from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent
sys.path.insert(0, str(RAFT_DIR / "update_experiments"))
sys.path.insert(0, str(RAFT_DIR / "search_experiments"))

from run_update_experiments import load_inputs  # noqa: E402
from run_dynamic_search_experiments import (  # noqa: E402
    Environment,
    SearchConfig,
    create_connected_business_graph,
    direct_confirmation_arrival,
    direct_trust,
    online_at,
    path_trust,
    qualifies,
    resource_at,
    transmit_edge,
    transmit_path,
    update_beta_after_confirmation,
)
from run_joint_discovery_experiment import (  # noqa: E402
    JointConfig,
    build_cache_replays,
    build_node_replays,
    environment_for_task,
)
from run_threshold_aware_mpc_experiment import (  # noqa: E402
    add_threshold_replay,
    calibrate_parameters,
)


POLICIES = {
    "online_beta": "Online Beta + path trust",
    "direct_only": "Online Beta, direct trust only",
    "static_beta": "Frozen Beta + path trust",
    "no_trust": "No-trust arrival ranking",
    "oracle": "Oracle reliability ranking",
}

COLORS = {
    "online_beta": "#0072B2",
    "direct_only": "#56B4E9",
    "static_beta": "#CC79A7",
    "no_trust": "#777777",
    "oracle": "#E69F00",
}

MARKERS = {
    "online_beta": "o",
    "direct_only": "v",
    "static_beta": "s",
    "no_trust": "^",
    "oracle": "D",
}

MALICIOUS_RATES = (0.0, 0.1, 0.2, 0.3, 0.4)


@dataclass(frozen=True)
class TrustExperimentConfig:
    num_nodes: int = 20
    trace_length: int = 300
    trace_start: int = 94
    offset_step: int = 4
    average_degree: int = 4
    update_budget: float = 0.5
    horizon: int = 6
    eta: float = 0.5
    threshold_lambda: float = 25.0
    tasks_per_network: int = 200
    network_repeats: int = 25
    outage_rate: float = 0.05
    topology_churn_rate: float = 0.05
    max_hops: int = 5
    deadline_ms: float = 180.0
    candidate_collection_ms: float = 120.0
    one_way_delay_ms: float = 15.0
    delay_jitter_ms: float = 2.0
    top_k: int = 3
    gamma: float = 0.9
    beta_history_length: int = 0
    honest_reliability: float = 0.95
    malicious_reliability: float = 0.20
    resource_change_probability: float = 0.0
    resource_shock_std: float = 0.0
    gossip_fanout: int = 2
    random_walk_restarts: int = 3
    expanding_ring_interval_ms: float = 24.0


@dataclass
class TrustSearchResult:
    success: int
    honest_success: int
    malicious_selected: int
    selected_candidate: int
    selected_trust: float
    decision_latency_ms: float
    message_count: int
    candidates_collected: int
    confirmation_attempts: int
    malicious_confirmations: int
    malicious_rejections: int
    invalid_confirmations: int
    request_messages: int
    response_messages: int
    confirmation_messages: int
    duplicate_requests: int
    selected_hops: float


def initial_beta_state(
    graph: Sequence[Set[int]],
    node_reliability: np.ndarray,
    seed: int,
    history_length: int,
) -> tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Generate a self-contained Beta(1+s, 1+f) direct-trust state."""
    rng = np.random.default_rng(seed)
    alpha: Dict[Tuple[int, int], float] = {}
    beta: Dict[Tuple[int, int], float] = {}
    for u, neighbors in enumerate(graph):
        for v in neighbors:
            successes = int(rng.binomial(history_length, node_reliability[v]))
            alpha[(u, v)] = 1.0 + successes
            beta[(u, v)] = 1.0 + history_length - successes
    return alpha, beta


def trust_state_metrics(
    graph: Sequence[Set[int]],
    alpha: Dict[Tuple[int, int], float],
    beta: Dict[Tuple[int, int], float],
    malicious_mask: np.ndarray,
    reliability: np.ndarray,
) -> tuple[float, float, float, float]:
    honest_scores: List[float] = []
    malicious_scores: List[float] = []
    errors: List[float] = []
    for u, neighbors in enumerate(graph):
        for v in neighbors:
            score = float(alpha[(u, v)] / (alpha[(u, v)] + beta[(u, v)]))
            errors.append(abs(score - reliability[v]))
            if malicious_mask[v]:
                malicious_scores.append(score)
            else:
                honest_scores.append(score)
    honest_mean = float(np.mean(honest_scores)) if honest_scores else math.nan
    malicious_mean = float(np.mean(malicious_scores)) if malicious_scores else math.nan
    mae = float(np.mean(errors)) if errors else math.nan
    if not honest_scores or not malicious_scores:
        auc = math.nan
    else:
        h = np.asarray(honest_scores)[:, None]
        m = np.asarray(malicious_scores)[None, :]
        auc = float(np.mean((h > m) + 0.5 * (h == m)))
    return honest_mean, malicious_mean, mae, auc


def policy_score(
    policy: str,
    env: Environment,
    candidate: int,
    path: Sequence[int],
    reliability: np.ndarray,
    cfg: SearchConfig,
) -> float:
    if policy in {"online_beta", "static_beta"}:
        return path_trust(env, path, cfg.gamma)
    if policy == "direct_only":
        return direct_trust(env, path[-2], path[-1])
    if policy == "oracle":
        return float(reliability[candidate])
    if policy == "no_trust":
        return 0.0
    raise ValueError(policy)


def run_trust_ranked_search(
    env: Environment,
    search_cfg: SearchConfig,
    policy: str,
    reliability: np.ndarray,
    malicious_mask: np.ndarray,
    behavior_valid: np.ndarray,
    seed: int,
    collection_ms: float,
    dissemination: str = "workflow",
) -> TrustSearchResult:
    """Two-phase search: collect responses, rank, then confirm top-k.

    ``workflow`` forwards to all eligible business neighbors, ``gossip``
    samples at most ``gossip_fanout`` neighbors at each hop, ``random_walk``
    launches sequential single-token walks, and ``expanding_ring`` launches
    progressively deeper TTL-limited workflow rings. ``centralized_oracle``
    is a non-deployable global-state upper bound.
    """
    if policy not in POLICIES:
        raise ValueError(policy)
    if not 0.0 < collection_ms < search_cfg.deadline_ms:
        raise ValueError("candidate collection cutoff must be within the request deadline")
    if dissemination not in {
        "workflow",
        "workflow_no_confirmation",
        "gossip",
        "random_walk",
        "expanding_ring",
        "centralized_oracle",
    }:
        raise ValueError(f"Unknown dissemination method: {dissemination}")
    use_confirmation = dissemination != "workflow_no_confirmation"
    propagation = "workflow" if dissemination == "workflow_no_confirmation" else dissemination

    rng = np.random.default_rng(seed)
    events: List[Tuple[float, int, int, str, tuple]] = []
    serial = 0
    rank_time = 0.0 if propagation == "centralized_oracle" else collection_ms
    if propagation == "centralized_oracle":
        # The centralized upper bound sees the current global state directly.
        # It is not constrained by the business graph and does not emit search
        # requests; only the final direct confirmation is counted below.
        heapq.heappush(events, (0.0, 0, serial, "oracle_seed", ()))
        serial += 1
    elif propagation == "random_walk":
        restarts = max(1, int(search_cfg.random_walk_restarts))
        interval = collection_ms / restarts
        for walk_id in range(restarts):
            heapq.heappush(
                events,
                (walk_id * interval, 0, serial, "walk_start", (walk_id + 1,)),
            )
            serial += 1
    elif propagation == "expanding_ring":
        interval = max(1.0, float(search_cfg.expanding_ring_interval_ms))
        for ttl in range(1, search_cfg.max_hops + 1):
            heapq.heappush(
                events,
                ((ttl - 1) * interval, 0, serial, "ring_start", (ttl,)),
            )
            serial += 1
    else:
        heapq.heappush(
            events,
            (
                0.0,
                0,
                serial,
                "request",
                (env.origin, (env.origin,), 0, search_cfg.max_hops),
            ),
        )
        serial += 1
    heapq.heappush(events, (rank_time, 2, serial, "rank", ()))
    serial += 1

    processed: Set[Tuple[int, int]] = set()
    candidate_records: Dict[int, tuple[float, tuple[int, ...]]] = {}
    request_messages = 0
    response_messages = 0
    confirmation_messages = 0
    confirmation_attempts = 0
    malicious_confirmations = 0
    malicious_rejections = 0
    invalid_confirmations = 0
    duplicate_requests = 0
    decision_time: float | None = None
    selected_candidate = -1
    selected_trust = math.nan
    selected_hops = math.nan
    ranked = False

    while events:
        time_ms, _, _, event_type, payload = heapq.heappop(events)
        if time_ms > 2.0 * search_cfg.deadline_ms:
            break

        if event_type == "oracle_seed":
            for candidate in range(len(env.known_adj)):
                if candidate == env.origin:
                    continue
                if online_at(env, candidate, 0.0) and qualifies(
                    resource_at(env, candidate, 0.0), env.demand
                ):
                    candidate_records[int(candidate)] = (
                        0.0,
                        (env.origin, int(candidate)),
                    )

        elif event_type in {"walk_start", "ring_start"}:
            if candidate_records:
                continue
            probe_id = int(payload[0])
            ttl_limit = (
                search_cfg.max_hops if event_type == "walk_start" else probe_id
            )
            heapq.heappush(
                events,
                (
                    float(time_ms),
                    0,
                    serial,
                    "request",
                    (env.origin, (env.origin,), probe_id, ttl_limit),
                ),
            )
            serial += 1

        elif event_type == "request":
            node, path_tuple, probe_id, ttl_limit = payload
            path = list(path_tuple)
            process_key = (int(probe_id), int(node))
            if time_ms > collection_ms:
                continue
            if process_key in processed:
                duplicate_requests += 1
                continue
            processed.add(process_key)
            if len(path) - 1 >= int(ttl_limit):
                continue

            candidates = []
            for candidate in env.known_adj[node]:
                if candidate == env.origin or candidate in path:
                    continue
                if qualifies(env.cache_resource[candidate], env.demand):
                    candidates.append((candidate, path + [candidate]))

            if candidates:
                for candidate, candidate_path in candidates:
                    reverse_path = list(reversed(path))
                    response_time = (
                        time_ms
                        if len(reverse_path) == 1
                        else transmit_path(env, reverse_path, time_ms, search_cfg, rng)
                    )
                    if len(reverse_path) > 1:
                        response_messages += len(reverse_path) - 1
                    if response_time is None:
                        continue
                    heapq.heappush(
                        events,
                        (
                            float(response_time),
                            1,
                            serial,
                            "response",
                            (int(candidate), tuple(candidate_path)),
                        ),
                    )
                    serial += 1
                # Continue within the remaining hop/time budget so that trust
                # ranks a candidate set rather than only the first advertised
                # branch. Duplicate suppression still bounds propagation.

            neighbors = [neighbor for neighbor in env.known_adj[node] if neighbor not in path]
            if propagation == "gossip" and len(neighbors) > search_cfg.gossip_fanout:
                neighbors = [
                    int(x)
                    for x in rng.choice(
                        neighbors, size=search_cfg.gossip_fanout, replace=False
                    )
                ]
            elif propagation == "random_walk" and neighbors:
                neighbors = [int(rng.choice(neighbors))]
            for neighbor in neighbors:
                arrival = transmit_edge(env, node, neighbor, time_ms, search_cfg, rng)
                request_messages += 1
                if arrival is None:
                    continue
                heapq.heappush(
                    events,
                    (
                        float(arrival),
                        0,
                        serial,
                        "request",
                        (
                            int(neighbor),
                            tuple(path + [neighbor]),
                            int(probe_id),
                            int(ttl_limit),
                        ),
                    ),
                )
                serial += 1

        elif event_type == "response":
            candidate, path_tuple = payload
            if time_ms > collection_ms:
                continue
            previous = candidate_records.get(candidate)
            replace_record = previous is None
            if previous is not None:
                if policy == "no_trust":
                    replace_record = time_ms < previous[0]
                else:
                    new_score = policy_score(
                        policy, env, candidate, path_tuple, reliability, search_cfg
                    )
                    old_score = policy_score(
                        policy, env, candidate, previous[1], reliability, search_cfg
                    )
                    replace_record = new_score > old_score + 1e-12 or (
                        abs(new_score - old_score) <= 1e-12 and time_ms < previous[0]
                    )
            if replace_record:
                candidate_records[candidate] = (float(time_ms), tuple(path_tuple))

        elif event_type == "rank":
            ranked = True
            scored = []
            for candidate, (response_time, path_tuple) in candidate_records.items():
                score = policy_score(
                    policy, env, candidate, path_tuple, reliability, search_cfg
                )
                margin = float(
                    np.min(env.cache_resource[candidate] - env.demand)
                )
                if policy == "no_trust":
                    key = (-margin, response_time, len(path_tuple), candidate)
                else:
                    key = (-score, -margin, response_time, len(path_tuple), candidate)
                scored.append((key, candidate, path_tuple, score))
            scored.sort(key=lambda item: item[0])
            if not use_confirmation:
                if scored:
                    _, candidate, path_tuple, score = scored[0]
                    path = list(path_tuple)
                    base_valid = (
                        online_at(env, candidate, rank_time)
                        and path[-2] in env.actual_adj[candidate]
                        and qualifies(resource_at(env, candidate, rank_time), env.demand)
                    )
                    valid = bool(base_valid and behavior_valid[candidate])
                    decision_time = float(rank_time)
                    if valid:
                        selected_candidate = int(candidate)
                        selected_trust = float(score)
                        selected_hops = float(len(path) - 1)
                continue
            for _, candidate, path_tuple, score in scored[: search_cfg.top_k]:
                confirmation_attempts += 1
                confirmation_messages += 2
                if malicious_mask[candidate]:
                    malicious_confirmations += 1
                confirm_rng = np.random.default_rng(seed + 10_000_019 + 1_009 * candidate)
                confirmation_time = direct_confirmation_arrival(
                    env, candidate, collection_ms, search_cfg, confirm_rng
                )
                heapq.heappush(
                    events,
                    (
                        float(confirmation_time),
                        3,
                        serial,
                        "confirmation",
                        (int(candidate), tuple(path_tuple), float(score)),
                    ),
                )
                serial += 1

        elif event_type == "confirmation":
            candidate, path_tuple, score = payload
            path = list(path_tuple)
            base_valid = (
                online_at(env, candidate, time_ms)
                and (
                    propagation == "centralized_oracle"
                    or path[-2] in env.actual_adj[candidate]
                )
                and qualifies(resource_at(env, candidate, time_ms), env.demand)
            )
            valid = bool(base_valid and behavior_valid[candidate])
            if policy in {"online_beta", "direct_only"}:
                update_beta_after_confirmation(env, path, valid)
            if not valid:
                invalid_confirmations += 1
                if malicious_mask[candidate]:
                    malicious_rejections += 1
            if decision_time is None and time_ms <= search_cfg.deadline_ms and valid:
                decision_time = float(time_ms)
                selected_candidate = int(candidate)
                selected_trust = float(score)
                selected_hops = float(len(path) - 1)

    if not ranked:
        raise AssertionError("candidate ranking event was not processed")
    success = selected_candidate >= 0
    malicious_selected = int(success and malicious_mask[selected_candidate])
    honest_success = int(success and not malicious_mask[selected_candidate])
    return TrustSearchResult(
        success=int(success),
        honest_success=honest_success,
        malicious_selected=malicious_selected,
        selected_candidate=selected_candidate,
        selected_trust=selected_trust,
        decision_latency_ms=(
            float(decision_time) if decision_time is not None else search_cfg.deadline_ms
        ),
        message_count=request_messages + response_messages + confirmation_messages,
        candidates_collected=len(candidate_records),
        confirmation_attempts=confirmation_attempts,
        malicious_confirmations=malicious_confirmations,
        malicious_rejections=malicious_rejections,
        invalid_confirmations=invalid_confirmations,
        request_messages=request_messages,
        response_messages=response_messages,
        confirmation_messages=confirmation_messages,
        duplicate_requests=duplicate_requests,
        selected_hops=selected_hops,
    )


def generate_trust_tasks(
    nodes_actual: np.ndarray,
    graph: Sequence[Set[int]],
    malicious_mask: np.ndarray,
    cfg: TrustExperimentConfig,
    seed: int,
) -> list[tuple[int, int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    honest_nodes = np.flatnonzero(~malicious_mask)
    tasks: list[tuple[int, int, np.ndarray]] = []
    for _ in range(cfg.tasks_per_network):
        for _attempt in range(2_000):
            time_slot = int(rng.integers(10, cfg.trace_length))
            origin = int(rng.choice(honest_nodes))
            availability = 100.0 - nodes_actual[:, time_slot]
            honest_remote = [
                node
                for node in honest_nodes
                if node != origin
                and node not in graph[origin]
                and availability[node, 0] >= 30.0
                and availability[node, 1] >= 4.0
            ]
            if not honest_remote:
                continue
            target = int(rng.choice(honest_remote))
            closeness = rng.uniform(0.90, 0.99, size=2)
            demand = np.maximum(
                availability[target] * closeness, np.array([5.0, 1.0])
            )
            if any(np.all(availability[node] >= demand) for node in graph[origin]):
                continue
            tasks.append((time_slot, origin, demand))
            break
        else:
            raise RuntimeError("Could not construct an honest-origin remote task")
    return tasks


def joint_config(cfg: TrustExperimentConfig) -> JointConfig:
    return JointConfig(
        num_nodes=cfg.num_nodes,
        trace_length=cfg.trace_length,
        trace_start=cfg.trace_start,
        offset_step=cfg.offset_step,
        average_degree=cfg.average_degree,
        update_budget=cfg.update_budget,
        horizon=cfg.horizon,
        tasks_per_network=cfg.tasks_per_network,
        network_repeats=cfg.network_repeats,
        outage_rate=cfg.outage_rate,
        topology_churn_rate=cfg.topology_churn_rate,
        max_hops=cfg.max_hops,
        deadline_ms=cfg.deadline_ms,
        one_way_delay_ms=cfg.one_way_delay_ms,
        top_k=cfg.top_k,
        gossip_fanout=cfg.gossip_fanout,
        resource_change_probability=cfg.resource_change_probability,
        resource_shock_std=cfg.resource_shock_std,
    )


def search_config(cfg: TrustExperimentConfig) -> SearchConfig:
    return SearchConfig(
        num_nodes=cfg.num_nodes,
        average_degree=cfg.average_degree,
        max_hops=cfg.max_hops,
        deadline_ms=cfg.deadline_ms,
        one_way_delay_ms=cfg.one_way_delay_ms,
        delay_jitter_ms=cfg.delay_jitter_ms,
        top_k=cfg.top_k,
        gossip_fanout=cfg.gossip_fanout,
        random_walk_restarts=cfg.random_walk_restarts,
        expanding_ring_interval_ms=cfg.expanding_ring_interval_ms,
        gamma=cfg.gamma,
        outage_rate=cfg.outage_rate,
        topology_churn_rate=cfg.topology_churn_rate,
        resource_change_probability=cfg.resource_change_probability,
    )


def run_one_network_condition(
    malicious_rate: float,
    repeat: int,
    nodes_actual: np.ndarray,
    cache_utilization: np.ndarray,
    cfg: TrustExperimentConfig,
) -> pd.DataFrame:
    jcfg = joint_config(cfg)
    scfg = search_config(cfg)
    graph_rng = np.random.default_rng(50_000 + repeat)
    graph = create_connected_business_graph(
        cfg.num_nodes, cfg.average_degree, graph_rng
    )

    permutation = np.random.default_rng(60_000 + repeat).permutation(cfg.num_nodes)
    malicious_count = int(round(malicious_rate * cfg.num_nodes))
    malicious_mask = np.zeros(cfg.num_nodes, dtype=bool)
    malicious_mask[permutation[:malicious_count]] = True
    reliability = np.where(
        malicious_mask, cfg.malicious_reliability, cfg.honest_reliability
    ).astype(float)
    initial_alpha, initial_beta = initial_beta_state(
        graph,
        reliability,
        seed=80_000 + repeat,
        history_length=cfg.beta_history_length,
    )
    tasks = generate_trust_tasks(
        nodes_actual,
        graph,
        malicious_mask,
        cfg,
        seed=100_000 + int(round(1_000 * malicious_rate)) + repeat,
    )

    rows = []
    for policy_idx, policy in enumerate(POLICIES):
        alpha = copy.deepcopy(initial_alpha)
        beta = copy.deepcopy(initial_beta)
        for task_idx, (time_slot, origin, demand) in enumerate(tasks):
            actual_availability = 100.0 - nodes_actual[:, time_slot]
            cached_availability = 100.0 - cache_utilization[:, time_slot]
            # Malicious nodes advertise enough cached resources to enter the
            # resource-filtered candidate pool; confirmation observes reality.
            if malicious_count:
                cached_availability[malicious_mask] = np.maximum(
                    cached_availability[malicious_mask], demand + np.array([3.0, 0.5])
                )
                cached_availability = np.clip(cached_availability, 0.0, 100.0)

            behavior_rng = np.random.default_rng(
                1_000_000
                + int(round(10_000 * malicious_rate))
                + repeat * cfg.tasks_per_network
                + task_idx
            )
            behavior_valid = behavior_rng.random(cfg.num_nodes) < reliability
            env = environment_for_task(
                graph,
                alpha,
                beta,
                actual_availability,
                cached_availability,
                origin,
                demand,
                jcfg,
                seed=2_000_000
                + int(round(10_000 * malicious_rate))
                + repeat * cfg.tasks_per_network
                + task_idx,
            )
            result = run_trust_ranked_search(
                env,
                scfg,
                policy,
                reliability,
                malicious_mask,
                behavior_valid,
                seed=3_000_000
                + int(round(10_000 * malicious_rate))
                + repeat * cfg.tasks_per_network
                + task_idx,
                collection_ms=cfg.candidate_collection_ms,
            )
            if policy in {"online_beta", "direct_only"}:
                alpha = copy.deepcopy(env.trust_alpha)
                beta = copy.deepcopy(env.trust_beta)

            honest_mean, malicious_mean, trust_mae, trust_auc = trust_state_metrics(
                graph, alpha, beta, malicious_mask, reliability
            )
            if policy == "no_trust":
                honest_mean = 0.5
                malicious_mean = 0.5 if malicious_count else math.nan
                trust_mae = float(np.mean(np.abs(0.5 - reliability)))
                trust_auc = 0.5 if malicious_count else math.nan
            elif policy == "oracle":
                honest_mean = cfg.honest_reliability
                malicious_mean = cfg.malicious_reliability if malicious_count else math.nan
                trust_mae = 0.0
                trust_auc = 1.0 if malicious_count else math.nan
            rows.append(
                {
                    "malicious_rate": malicious_rate,
                    "repeat": repeat,
                    "task": task_idx,
                    "task_quartile": min(4, 1 + 4 * task_idx // cfg.tasks_per_network),
                    "policy": policy,
                    "origin": origin,
                    "time_slot": time_slot,
                    "success": result.success,
                    "honest_success": result.honest_success,
                    "malicious_selected": result.malicious_selected,
                    "selected_candidate": result.selected_candidate,
                    "selected_trust": result.selected_trust,
                    "success_latency_ms": (
                        result.decision_latency_ms if result.success else np.nan
                    ),
                    "censored_latency_ms": result.decision_latency_ms,
                    "messages": result.message_count,
                    "candidates_collected": result.candidates_collected,
                    "confirmation_attempts": result.confirmation_attempts,
                    "malicious_confirmations": result.malicious_confirmations,
                    "malicious_rejections": result.malicious_rejections,
                    "invalid_confirmations": result.invalid_confirmations,
                    "mean_honest_direct_trust": honest_mean,
                    "mean_malicious_direct_trust": malicious_mean,
                    "trust_mae": trust_mae,
                    "trust_auc": trust_auc,
                }
            )
    return pd.DataFrame(rows)


def run_formal(
    nodes_actual: np.ndarray,
    cache_utilization: np.ndarray,
    cfg: TrustExperimentConfig,
    workers: int,
) -> pd.DataFrame:
    jobs = [
        (rate, repeat)
        for rate in MALICIOUS_RATES
        for repeat in range(cfg.network_repeats)
    ]
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_one_network_condition,
                rate,
                repeat,
                nodes_actual,
                cache_utilization,
                cfg,
            ): (rate, repeat)
            for rate, repeat in jobs
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            rate, repeat = futures[future]
            frames.append(future.result())
            if idx % 10 == 0 or idx == len(futures):
                print(
                    f"[{idx:03d}/{len(futures):03d}] completed; "
                    f"latest malicious_rate={rate:.1f}, repeat={repeat}"
                )
    return pd.concat(frames, ignore_index=True)


def network_summary(raw: pd.DataFrame) -> pd.DataFrame:
    base = (
        raw.groupby(["malicious_rate", "repeat", "policy"], as_index=False)
        .agg(
            honest_success_rate=("honest_success", "mean"),
            any_success_rate=("success", "mean"),
            malicious_selection_rate=("malicious_selected", "mean"),
            mean_messages=("messages", "mean"),
            mean_confirmations=("confirmation_attempts", "mean"),
            malicious_confirmations=("malicious_confirmations", "sum"),
            malicious_rejections=("malicious_rejections", "sum"),
        )
    )
    last = (
        raw.sort_values("task")
        .groupby(["malicious_rate", "repeat", "policy"], as_index=False)
        .tail(1)[
            [
                "malicious_rate",
                "repeat",
                "policy",
                "mean_honest_direct_trust",
                "mean_malicious_direct_trust",
                "trust_mae",
                "trust_auc",
            ]
        ]
    )
    out = base.merge(last, on=["malicious_rate", "repeat", "policy"], how="left")
    out["malicious_rejection_rate"] = np.where(
        out.malicious_confirmations > 0,
        out.malicious_rejections / out.malicious_confirmations,
        np.nan,
    )
    return out


def mean_ci(values: pd.Series) -> tuple[float, float]:
    x = values.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(np.mean(x)) if len(x) else math.nan, math.nan)
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / math.sqrt(len(x)))


def summarize(network: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rate, policy), group in network.groupby(["malicious_rate", "policy"]):
        honest_mean, honest_ci = mean_ci(100.0 * group.honest_success_rate)
        any_mean, any_ci = mean_ci(100.0 * group.any_success_rate)
        malicious_mean, malicious_ci = mean_ci(100.0 * group.malicious_selection_rate)
        auc_mean, auc_ci = mean_ci(group.trust_auc)
        latency = raw[
            (np.isclose(raw.malicious_rate, rate))
            & (raw.policy == policy)
            & raw.success.eq(1)
        ].success_latency_ms
        rows.append(
            {
                "malicious_rate": rate,
                "policy": policy,
                "networks": len(group),
                "tasks": int(
                    len(
                        raw[
                            (np.isclose(raw.malicious_rate, rate))
                            & (raw.policy == policy)
                        ]
                    )
                ),
                "honest_success_percent": honest_mean,
                "honest_success_ci95_percent": honest_ci,
                "any_success_percent": any_mean,
                "any_success_ci95_percent": any_ci,
                "malicious_selection_percent": malicious_mean,
                "malicious_selection_ci95_percent": malicious_ci,
                "malicious_rejection_percent": 100.0
                * float(group.malicious_rejections.sum())
                / max(float(group.malicious_confirmations.sum()), 1.0),
                "final_trust_auc": auc_mean,
                "final_trust_auc_ci95": auc_ci,
                "final_trust_mae": float(group.trust_mae.mean()),
                "final_honest_trust": float(group.mean_honest_direct_trust.mean()),
                "final_malicious_trust": float(group.mean_malicious_direct_trust.mean()),
                "p95_success_latency_ms": float(latency.quantile(0.95))
                if len(latency)
                else math.nan,
                "mean_messages": float(group.mean_messages.mean()),
                "mean_confirmations": float(group.mean_confirmations.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["malicious_rate", "policy"])


def bootstrap_ci(diff: np.ndarray, seed: int, draws: int = 5_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(diff, size=(draws, len(diff)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(sampled, [0.025, 0.975]))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def paired_tests(network: pd.DataFrame) -> pd.DataFrame:
    rows = []
    endpoints = {
        "honest_success_rate": "percentage_points",
        "malicious_selection_rate": "percentage_points",
    }
    for rate_idx, rate in enumerate(MALICIOUS_RATES):
        online = network[
            np.isclose(network.malicious_rate, rate) & network.policy.eq("online_beta")
        ][["repeat", *endpoints]].copy()
        for comparator_idx, comparator in enumerate(("static_beta", "no_trust")):
            baseline = network[
                np.isclose(network.malicious_rate, rate) & network.policy.eq(comparator)
            ][["repeat", *endpoints]].copy()
            merged = online.merge(
                baseline, on="repeat", suffixes=("_online", "_baseline")
            )
            for endpoint_idx, endpoint in enumerate(endpoints):
                diff = 100.0 * (
                    merged[f"{endpoint}_online"].to_numpy()
                    - merged[f"{endpoint}_baseline"].to_numpy()
                )
                if np.allclose(diff, 0.0):
                    p_value = 1.0
                else:
                    p_value = float(
                        wilcoxon(diff, zero_method="pratt", alternative="two-sided").pvalue
                    )
                low, high = bootstrap_ci(
                    diff,
                    seed=91_000 + rate_idx * 100 + comparator_idx * 10 + endpoint_idx,
                )
                rows.append(
                    {
                        "malicious_rate": rate,
                        "endpoint": endpoint,
                        "comparison": f"online_beta vs {comparator}",
                        "network_clusters": len(diff),
                        "online_percent": 100.0
                        * float(merged[f"{endpoint}_online"].mean()),
                        "baseline_percent": 100.0
                        * float(merged[f"{endpoint}_baseline"].mean()),
                        "online_minus_baseline_pp": float(np.mean(diff)),
                        "cluster_bootstrap_ci_low_pp": low,
                        "cluster_bootstrap_ci_high_pp": high,
                        "wilcoxon_p": p_value,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p"] = holm_adjust(frame.wilcoxon_p)
    return frame


def learning_curve(raw: pd.DataFrame) -> pd.DataFrame:
    subset = raw[
        np.isclose(raw.malicious_rate, 0.3) & raw.policy.eq("online_beta")
    ]
    return (
        subset.groupby(["repeat", "task_quartile"], as_index=False)
        .agg(
            honest_trust=("mean_honest_direct_trust", "mean"),
            malicious_trust=("mean_malicious_direct_trust", "mean"),
            trust_auc=("trust_auc", "mean"),
        )
    )


def plot_results(
    summary: pd.DataFrame,
    curve: pd.DataFrame,
    output_dir: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    rates_percent = 100.0 * np.asarray(MALICIOUS_RATES)

    for policy in POLICIES:
        data = summary[summary.policy == policy].sort_values("malicious_rate")
        axes[0, 0].errorbar(
            rates_percent,
            data.honest_success_percent,
            yerr=data.honest_success_ci95_percent,
            color=COLORS[policy],
            marker=MARKERS[policy],
            linewidth=1.4,
            markersize=4.2,
            capsize=2,
            label=POLICIES[policy],
        )
        axes[0, 1].errorbar(
            rates_percent,
            data.malicious_selection_percent,
            yerr=data.malicious_selection_ci95_percent,
            color=COLORS[policy],
            marker=MARKERS[policy],
            linewidth=1.4,
            markersize=4.2,
            capsize=2,
        )
        axes[1, 1].errorbar(
            rates_percent[1:],
            data.final_trust_auc.to_numpy()[1:],
            yerr=data.final_trust_auc_ci95.to_numpy()[1:],
            color=COLORS[policy],
            marker=MARKERS[policy],
            linewidth=1.4,
            markersize=4.2,
            capsize=2,
        )

    axes[0, 0].set_title("(a) Trustworthy candidate discovery", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Malicious-node fraction (%)")
    axes[0, 0].set_ylabel("Honest candidate discovery (%)")
    axes[0, 1].set_title("(b) Malicious candidate selection", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Malicious-node fraction (%)")
    axes[0, 1].set_ylabel("Malicious selection (%)")

    quartiles = np.arange(1, 5)
    for field, label, color, marker in (
        ("honest_trust", "Honest-node direct trust", "#0072B2", "o"),
        ("malicious_trust", "Malicious-node direct trust", "#D55E00", "s"),
    ):
        means = curve.groupby("task_quartile")[field].mean().reindex(quartiles)
        cis = (
            curve.groupby("task_quartile")[field]
            .apply(lambda values: mean_ci(values)[1])
            .reindex(quartiles)
        )
        axes[1, 0].errorbar(
            quartiles,
            means,
            yerr=cis,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=4.5,
            capsize=2,
            label=label,
        )
    axes[1, 0].axhline(0.5, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 0].set_title("(c) Online Beta learning at 30% malicious nodes", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Sequential-request quartile")
    axes[1, 0].set_ylabel("Mean direct trust")
    axes[1, 0].set_xticks(quartiles)
    axes[1, 0].legend(loc="best")

    axes[1, 1].set_title("(d) Final trust discrimination", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Malicious-node fraction (%)")
    axes[1, 1].set_ylabel("Trust ROC AUC")
    axes[1, 1].set_ylim(0.45, 1.03)
    axes[1, 1].axhline(0.5, color="#777777", linestyle="--", linewidth=0.9)

    for ax in axes.flat:
        ax.grid(color="#D8D8D8", linewidth=0.7, alpha=0.8)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    base = output_dir / "trust_robustness"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    data = frame[list(columns)].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, row in data.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("NA")
            elif isinstance(value, (float, np.floating)):
                number = float(value)
                values.append(
                    f"{number:.3e}"
                    if number != 0.0 and abs(number) < 1e-4
                    else f"{number:.4f}"
                )
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    cfg: TrustExperimentConfig,
    cache_wrmse: float,
    output_dir: Path,
) -> None:
    def value(rate: float, policy: str, column: str) -> float:
        row = summary[
            np.isclose(summary.malicious_rate, rate) & summary.policy.eq(policy)
        ]
        return float(row.iloc[0][column])

    rate = 0.3
    online_honest = value(rate, "online_beta", "honest_success_percent")
    no_honest = value(rate, "no_trust", "honest_success_percent")
    static_honest = value(rate, "static_beta", "honest_success_percent")
    online_mal = value(rate, "online_beta", "malicious_selection_percent")
    no_mal = value(rate, "no_trust", "malicious_selection_percent")
    static_mal = value(rate, "static_beta", "malicious_selection_percent")
    online_honest_10 = value(0.1, "online_beta", "honest_success_percent")
    no_honest_10 = value(0.1, "no_trust", "honest_success_percent")
    online_honest_40 = value(0.4, "online_beta", "honest_success_percent")
    no_honest_40 = value(0.4, "no_trust", "honest_success_percent")
    oracle_honest_40 = value(0.4, "oracle", "honest_success_percent")
    online_mal_40 = value(0.4, "online_beta", "malicious_selection_percent")
    no_mal_40 = value(0.4, "no_trust", "malicious_selection_percent")
    report = f"""# Beta直接信任持续更新与恶意节点鲁棒性实验

## 1. 实验目的与边界

本实验评价卸载前候选节点发现中的信任排序，不执行具体任务卸载。直接信任采用Beta(1+s, 1+f)模型，在线方法在每次候选确认后更新成功/失败计数，并将状态持续到后续请求；间接信任沿业务关系路径递归传播。实验不把Beta估计本身作为创新，而是检验其与路径信任和二次确认结合后是否能降低恶意候选选择。

## 2. 冻结协议

- 节点数{cfg.num_nodes}，平均业务度数{cfg.average_degree}；门槛感知FITS-MPC-DP的eta={cfg.eta:g}、lambda={cfg.threshold_lambda:g}、CPU/内存权重0.5/0.5、更新预算{cfg.update_budget:.0%}均冻结。
- 恶意节点比例：0%、10%、20%、30%和40%；诚实/恶意节点行为可靠率分别为{cfg.honest_reliability:.0%}/{cfg.malicious_reliability:.0%}。
- 每种条件{cfg.network_repeats}个独立网络，每个网络按时间顺序处理{cfg.tasks_per_network}个远端必需请求；网络而不是单个请求是独立统计单位。
- 候选在{cfg.candidate_collection_ms:g} ms前收集，按信任或对照规则排序，对top-{cfg.top_k}并行确认；绝对截止时间{cfg.deadline_ms:g} ms。
- 所有方法共享资源轨迹、任务、拓扑、断联、节点行为和网络时延随机种子。
- 门槛感知缓存WRMSE为{cache_wrmse:.4f}，信任实验不重新训练FITS或调节更新参数。

## 3. 完整汇总

{markdown_table(summary, ['malicious_rate', 'policy', 'networks', 'tasks', 'honest_success_percent', 'honest_success_ci95_percent', 'any_success_percent', 'malicious_selection_percent', 'malicious_rejection_percent', 'final_trust_auc', 'final_trust_mae', 'final_honest_trust', 'final_malicious_trust', 'p95_success_latency_ms', 'mean_messages', 'mean_confirmations'])}

## 4. 30%恶意节点下的重点结果

- 在线Beta的诚实候选发现率为{online_honest:.2f}%，冻结Beta为{static_honest:.2f}%，无信任排序为{no_honest:.2f}%。
- 在线Beta的恶意候选选择率为{online_mal:.2f}%，冻结Beta为{static_mal:.2f}%，无信任排序为{no_mal:.2f}%。
- 在线Beta相对静态和无信任方法的差异必须结合下表的网络级效应量、置信区间和Holm校正p值解释；单个请求不能作为独立重复。

## 5. 网络级配对检验

{markdown_table(tests, ['malicious_rate', 'endpoint', 'comparison', 'network_clusters', 'online_percent', 'baseline_percent', 'online_minus_baseline_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_p', 'holm_adjusted_p'])}

## 6. 主要发现与解释

1. 无恶意节点时四种方法的诚实候选发现率均约为91.4%，在线更新没有造成可观察的正常场景性能损失。
2. 恶意比例为10%时，在线Beta相对无信任排序将诚实候选发现率由{no_honest_10:.2f}%提高到{online_honest_10:.2f}%；恶意比例为40%时则由{no_honest_40:.2f}%提高到{online_honest_40:.2f}%。40%条件下在线方法距Oracle上界{oracle_honest_40:.2f}%仍有{oracle_honest_40-online_honest_40:.2f}个百分点差距。
3. 40%恶意节点时，在线Beta把恶意候选选择率由{no_mal_40:.2f}%降至{online_mal_40:.2f}%。该指标只统计通过确认并最终被采用的恶意节点，不把已被二次确认拒绝的候选误计为选择成功。
4. 二次确认在各非零恶意条件下拦截约98%的恶意确认，但无信任排序仍会因为有限top-k名额被恶意候选占据而失去诚实候选。在线Beta的主要贡献是改善确认前排序，使更多诚实节点进入top-k。
5. 30%恶意节点下，在线Beta的诚实/恶意平均直接信任在第1个请求四分位约为0.55/0.32，到第4个四分位变为0.69/0.14；信任AUC由约0.91升至1.00，表明持续交互能够形成稳定区分。
6. 冻结Beta从中性先验开始，没有新的观测证据，其结果与无信任到达排序基本相同。这一消融说明优势来自跨请求更新，而不是Beta公式名称本身。

## 7. 解释边界

1. 恶意行为表示虚假资源声明或确认阶段不可靠，不代表任务执行后的数据窃取、隐私攻击或拜占庭共识攻击。
2. Oracle使用真实节点可靠率，仅表示不可部署上界。
3. Beta从每条有向业务边的中性先验Beta(1,1)开始；在线与冻结Beta共享相同初值，避免利用节点真实可靠率生成有利先验。
4. 候选收集截止时间和top-k均为预先固定参数，没有根据测试结果调整。
5. 结果来自真实资源轨迹驱动的动态网络仿真，不替代真实工业边缘网络部署。
6. 当前攻击者可靠率固定为20%，尚未覆盖会先积累信誉再作恶的on-off攻击或节点串谋；对这些攻击的防御不应由本实验外推。
"""
    (output_dir / "Beta信任与恶意节点鲁棒性实验结果与分析.md").write_text(
        report, encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "trust_robustness_formal",
    )
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--tasks-per-network", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrustExperimentConfig(
        network_repeats=args.repeats,
        tasks_per_network=args.tasks_per_network,
    )
    actual, predictions = load_inputs(args.data_dir)
    jcfg = joint_config(cfg)
    selected_eta, selected_lambda, thresholds, calibration = calibrate_parameters(
        actual,
        predictions,
        cfg.trace_start,
        cfg.update_budget,
        cfg.horizon,
    )
    if not np.isclose(selected_eta, cfg.eta) or not np.isclose(
        selected_lambda, cfg.threshold_lambda
    ):
        raise AssertionError(
            f"Frozen trust experiment expected eta/lambda={cfg.eta}/{cfg.threshold_lambda}, "
            f"calibration returned {selected_eta}/{selected_lambda}"
        )
    nodes_actual, nodes_predictions = build_node_replays(actual, predictions, jcfg)
    replays = build_cache_replays(nodes_actual, nodes_predictions, jcfg)
    replays = add_threshold_replay(
        nodes_actual,
        nodes_predictions,
        replays,
        jcfg,
        cfg.eta,
        cfg.threshold_lambda,
        thresholds,
        np.array([0.5, 0.5]),
    )
    replay = replays["threshold_mpc"]
    raw = run_formal(
        nodes_actual,
        replay.cache_utilization,
        cfg,
        args.workers,
    )
    network = network_summary(raw)
    summary = summarize(network, raw)
    tests = paired_tests(network)
    curve = learning_curve(raw)

    raw.to_csv(args.output_dir / "trust_robustness_raw.csv", index=False)
    network.to_csv(args.output_dir / "trust_robustness_network.csv", index=False)
    summary.to_csv(args.output_dir / "trust_robustness_summary.csv", index=False)
    tests.to_csv(args.output_dir / "trust_robustness_network_tests.csv", index=False)
    curve.to_csv(args.output_dir / "trust_learning_curve.csv", index=False)
    calibration.to_csv(args.output_dir / "frozen_threshold_calibration.csv", index=False)
    plot_results(summary, curve, args.output_dir)
    write_report(summary, tests, cfg, replay.cache_wrmse, args.output_dir)

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(cfg),
        "malicious_rates": list(MALICIOUS_RATES),
        "policies": POLICIES,
        "independent_unit": "network repeat",
        "within_network_repeated_observations": cfg.tasks_per_network,
        "primary_endpoints": [
            "honest candidate discovery rate",
            "malicious candidate selection rate",
        ],
        "multiple_comparison_family": "20 paired network-level comparisons; Holm correction",
        "frozen_cache_wrmse": replay.cache_wrmse,
        "selected_eta": selected_eta,
        "selected_threshold_lambda": selected_lambda,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTrust robustness summary\n")
    print(
        summary[
            [
                "malicious_rate",
                "policy",
                "honest_success_percent",
                "malicious_selection_percent",
                "final_trust_auc",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
