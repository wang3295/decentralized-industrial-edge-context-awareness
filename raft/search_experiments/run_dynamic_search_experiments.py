from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "workflow_confirm": "Workflow + confirmation",
    "workflow_no_confirm": "Workflow, no confirmation",
    "gossip_confirm": "Gossip + confirmation",
}

COLORS = {
    "workflow_confirm": "#0072B2",
    "workflow_no_confirm": "#D55E00",
    "gossip_confirm": "#009E73",
}

MARKERS = {
    "workflow_confirm": "o",
    "workflow_no_confirm": "s",
    "gossip_confirm": "^",
}


@dataclass(frozen=True)
class SearchConfig:
    num_nodes: int = 100
    average_degree: int = 4
    max_hops: int = 5
    deadline_ms: float = 180.0
    one_way_delay_ms: float = 15.0
    delay_jitter_ms: float = 2.0
    top_k: int = 3
    gossip_fanout: int = 2
    random_walk_restarts: int = 3
    expanding_ring_interval_ms: float = 24.0
    gamma: float = 0.9
    outage_rate: float = 0.10
    topology_churn_rate: float = 0.10
    resource_change_probability: float = 0.30
    resource_shock_std: float = 11.0


@dataclass
class Environment:
    known_adj: List[Set[int]]
    actual_adj: List[Set[int]]
    cache_resource: np.ndarray
    resource_pre: np.ndarray
    resource_post: np.ndarray
    resource_change_time: np.ndarray
    outage_start: np.ndarray
    outage_end: np.ndarray
    trust_alpha: Dict[Tuple[int, int], float]
    trust_beta: Dict[Tuple[int, int], float]
    origin: int
    demand: np.ndarray


@dataclass
class SearchResult:
    method: str
    success: int
    decision_latency_ms: float
    message_count: int
    request_messages: int
    response_messages: int
    confirmation_messages: int
    candidates_in_time: int
    confirmation_attempts: int
    stale_or_invalid_candidates: int
    cached_after_decision: int
    selected_hops: float
    selected_trust: float
    selected_candidate: int


def add_undirected(adj: List[Set[int]], u: int, v: int) -> None:
    if u == v:
        return
    adj[u].add(v)
    adj[v].add(u)


def remove_undirected(adj: List[Set[int]], u: int, v: int) -> None:
    adj[u].discard(v)
    adj[v].discard(u)


def edge_list(adj: Sequence[Set[int]]) -> List[Tuple[int, int]]:
    return [(u, v) for u, nbrs in enumerate(adj) for v in nbrs if u < v]


def create_connected_business_graph(n: int, average_degree: int, rng: np.random.Generator) -> List[Set[int]]:
    """Create a connected sparse graph with an approximately fixed average degree."""
    adj = [set() for _ in range(n)]
    order = rng.permutation(n)
    for idx in range(1, n):
        parent = int(order[int(rng.integers(0, idx))])
        add_undirected(adj, int(order[idx]), parent)

    target_edges = max(n - 1, int(round(n * average_degree / 2)))
    while len(edge_list(adj)) < target_edges:
        u, v = (int(x) for x in rng.choice(n, size=2, replace=False))
        add_undirected(adj, u, v)
    return adj


def churn_graph(known_adj: Sequence[Set[int]], rate: float, rng: np.random.Generator) -> List[Set[int]]:
    """Replace a fraction of business links; nodes still hold the pre-change neighbor view."""
    actual = [set(nbrs) for nbrs in known_adj]
    edges = edge_list(actual)
    replace_count = min(len(edges), int(round(len(edges) * rate)))
    if replace_count == 0:
        return actual

    selected = rng.choice(len(edges), size=replace_count, replace=False)
    for idx in selected:
        u, v = edges[int(idx)]
        remove_undirected(actual, u, v)

    target_edges = len(edges)
    attempts = 0
    while len(edge_list(actual)) < target_edges and attempts < target_edges * 100:
        u, v = (int(x) for x in rng.choice(len(actual), size=2, replace=False))
        add_undirected(actual, u, v)
        attempts += 1
    return actual


def beta_reputation(
    adj: Sequence[Set[int]], rng: np.random.Generator, history_length: int = 20
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Self-contained direct trust using a Beta(1+s, 1+f) reputation model."""
    n = len(adj)
    node_reliability = rng.beta(8.0, 2.0, size=n)
    alpha: Dict[Tuple[int, int], float] = {}
    beta: Dict[Tuple[int, int], float] = {}
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            successes = int(rng.binomial(history_length, node_reliability[v]))
            alpha[(u, v)] = 1.0 + successes
            beta[(u, v)] = 1.0 + history_length - successes
    return alpha, beta


def generate_environment(seed: int, cfg: SearchConfig) -> Environment:
    ss = np.random.SeedSequence(seed)
    rng_graph, rng_churn, rng_resource, rng_query, rng_dynamic, rng_outage, rng_trust = [
        np.random.default_rng(s) for s in ss.spawn(7)
    ]
    known = create_connected_business_graph(cfg.num_nodes, cfg.average_degree, rng_graph)
    actual = churn_graph(known, cfg.topology_churn_rate, rng_churn)

    cache = np.clip(rng_resource.normal(50.0, 15.0, size=(cfg.num_nodes, 2)), 5.0, 95.0)
    mean_reversion = 0.88 * cache + 0.12 * 50.0
    pre = np.clip(mean_reversion + rng_dynamic.normal(0.0, cfg.resource_shock_std, cache.shape), 0.0, 100.0)
    changes = rng_dynamic.random(cfg.num_nodes) < cfg.resource_change_probability
    post = pre.copy()
    # Adverse availability event: a new workload consumes free CPU/memory while
    # the search is in flight. This is a one-sided robustness stress test.
    post[changes] = np.clip(
        pre[changes]
        - np.abs(rng_dynamic.normal(0.0, cfg.resource_shock_std, size=(int(changes.sum()), 2))),
        0.0,
        100.0,
    )
    # Fault/resource event distributions are held fixed in absolute time so that
    # changing the deadline is a genuine deadline sensitivity experiment.
    reference_deadline_ms = 180.0
    change_time = rng_dynamic.uniform(
        0.15 * reference_deadline_ms, 0.85 * reference_deadline_ms, cfg.num_nodes
    )

    outage_mask = rng_outage.random(cfg.num_nodes) < cfg.outage_rate
    outage_start = np.full(cfg.num_nodes, np.inf)
    outage_end = np.full(cfg.num_nodes, np.inf)
    outage_start[outage_mask] = rng_outage.uniform(
        0.0, 0.60 * reference_deadline_ms, int(outage_mask.sum())
    )
    outage_end[outage_mask] = outage_start[outage_mask] + rng_outage.uniform(
        0.25 * reference_deadline_ms, 1.20 * reference_deadline_ms, int(outage_mask.sum())
    )

    # The search experiment is conditioned on the local-neighbor phase failing,
    # which is exactly when the paper's multi-hop mechanism is invoked.
    origin = 0
    demand = np.array([65.0, 65.0])
    found = False
    for _ in range(1000):
        candidate_origin = int(rng_query.integers(0, cfg.num_nodes))
        candidate_demand = rng_query.uniform(56.0, 76.0, size=2)
        direct_ok = any(np.all(cache[v] >= candidate_demand) for v in known[candidate_origin])
        global_ok = any(
            v != candidate_origin
            and v not in known[candidate_origin]
            and np.all(cache[v] >= candidate_demand)
            for v in range(cfg.num_nodes)
        )
        if not direct_ok and global_ok:
            origin = candidate_origin
            demand = candidate_demand
            found = True
            break
    if not found:
        raise RuntimeError("Could not construct a multi-hop search query")

    outage_start[origin] = np.inf
    outage_end[origin] = np.inf
    alpha, beta = beta_reputation(known, rng_trust)
    return Environment(
        known_adj=known,
        actual_adj=actual,
        cache_resource=cache,
        resource_pre=pre,
        resource_post=post,
        resource_change_time=change_time,
        outage_start=outage_start,
        outage_end=outage_end,
        trust_alpha=alpha,
        trust_beta=beta,
        origin=origin,
        demand=demand,
    )


def direct_trust(env: Environment, u: int, v: int) -> float:
    a = env.trust_alpha[(u, v)]
    b = env.trust_beta[(u, v)]
    return a / (a + b)


def path_trust(env: Environment, path: Sequence[int], gamma: float) -> float:
    if len(path) < 2:
        return 1.0
    value = direct_trust(env, path[0], path[1])
    for idx in range(1, len(path) - 1):
        value = gamma * math.sqrt(value * direct_trust(env, path[idx], path[idx + 1]))
    return float(value)


def next_online_time(env: Environment, node: int, time_ms: float) -> float:
    start = env.outage_start[node]
    end = env.outage_end[node]
    if start <= time_ms < end:
        return float(end)
    return time_ms


def online_at(env: Environment, node: int, time_ms: float) -> bool:
    return not (env.outage_start[node] <= time_ms < env.outage_end[node])


def resource_at(env: Environment, node: int, time_ms: float) -> np.ndarray:
    if time_ms >= env.resource_change_time[node]:
        return env.resource_post[node]
    return env.resource_pre[node]


def qualifies(resource: np.ndarray, demand: np.ndarray) -> bool:
    return bool(np.all(resource >= demand))


def draw_delay(cfg: SearchConfig, rng: np.random.Generator) -> float:
    return float(max(1.0, rng.normal(cfg.one_way_delay_ms, cfg.delay_jitter_ms)))


def transmit_edge(
    env: Environment,
    u: int,
    v: int,
    start_ms: float,
    cfg: SearchConfig,
    rng: np.random.Generator,
) -> Optional[float]:
    if v not in env.actual_adj[u]:
        return None
    depart = next_online_time(env, u, start_ms)
    arrival = depart + draw_delay(cfg, rng)
    arrival = next_online_time(env, v, arrival)
    return float(arrival)


def transmit_path(
    env: Environment,
    path: Sequence[int],
    start_ms: float,
    cfg: SearchConfig,
    rng: np.random.Generator,
) -> Optional[float]:
    time_ms = start_ms
    for u, v in zip(path[:-1], path[1:]):
        next_time = transmit_edge(env, u, v, time_ms, cfg, rng)
        if next_time is None:
            return None
        time_ms = next_time
    return time_ms


def direct_confirmation_arrival(
    env: Environment, candidate: int, start_ms: float, cfg: SearchConfig, rng: np.random.Generator
) -> float:
    # Candidate address has been returned, so confirmation uses a direct IP-level
    # request/reply. Outage delays the response, but does not consume workflow hops.
    send_time = next_online_time(env, env.origin, start_ms)
    at_candidate = send_time + draw_delay(cfg, rng)
    at_candidate = next_online_time(env, candidate, at_candidate)
    return_time = at_candidate + draw_delay(cfg, rng)
    return next_online_time(env, env.origin, return_time)


def update_beta_after_confirmation(env: Environment, path: Sequence[int], success: bool) -> None:
    if len(path) < 2:
        return
    edge = (path[-2], path[-1])
    if success:
        env.trust_alpha[edge] += 1.0
    else:
        env.trust_beta[edge] += 1.0


def run_search(method: str, source_env: Environment, cfg: SearchConfig, seed: int) -> SearchResult:
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method: {method}")
    env = copy.deepcopy(source_env)
    rng = np.random.default_rng(seed)
    use_confirmation = method != "workflow_no_confirm"
    gossip = method == "gossip_confirm"
    horizon = 2.0 * cfg.deadline_ms

    # Event tuple: time, priority, trust_order, serial, type, payload.
    events: List[Tuple[float, int, float, int, str, tuple]] = []
    serial = 0
    heapq.heappush(events, (0.0, 0, 0.0, serial, "request", (env.origin, (env.origin,))))
    serial += 1
    processed: Set[int] = set()
    returned_candidates: Set[int] = set()
    confirmation_candidates: Set[int] = set()

    request_messages = 0
    response_messages = 0
    confirmation_messages = 0
    candidates_in_time = 0
    confirmation_attempts = 0
    stale_or_invalid = 0
    cached_after_decision = 0
    decision_time: Optional[float] = None
    success = False
    selected_hops = math.nan
    selected_trust = math.nan
    selected_candidate = -1

    while events:
        time_ms, _, _, _, event_type, payload = heapq.heappop(events)
        if time_ms > horizon:
            break

        if event_type == "request":
            node, path_tuple = payload
            path = list(path_tuple)
            if time_ms > cfg.deadline_ms or node in processed:
                continue
            processed.add(node)
            depth = len(path) - 1
            if depth >= cfg.max_hops:
                continue

            candidates = []
            for candidate in env.known_adj[node]:
                if candidate == env.origin or candidate in path:
                    continue
                if qualifies(env.cache_resource[candidate], env.demand):
                    candidate_path = path + [candidate]
                    candidates.append((candidate, candidate_path, path_trust(env, candidate_path, cfg.gamma)))

            if candidates:
                for candidate, candidate_path, trust_value in candidates:
                    reverse_path = list(reversed(path))
                    if len(reverse_path) == 1:
                        response_time = time_ms
                    else:
                        response_time = transmit_path(env, reverse_path, time_ms, cfg, rng)
                        response_messages += len(reverse_path) - 1
                    if response_time is None:
                        continue
                    heapq.heappush(
                        events,
                        (
                            response_time,
                            1,
                            -trust_value,
                            serial,
                            "response",
                            (candidate, tuple(candidate_path), trust_value),
                        ),
                    )
                    serial += 1
                continue

            neighbors = [v for v in env.known_adj[node] if v not in path]
            if gossip and len(neighbors) > cfg.gossip_fanout:
                neighbors = [int(x) for x in rng.choice(neighbors, size=cfg.gossip_fanout, replace=False)]
            for neighbor in neighbors:
                arrival = transmit_edge(env, node, neighbor, time_ms, cfg, rng)
                request_messages += 1
                if arrival is None:
                    continue
                heapq.heappush(
                    events,
                    (arrival, 0, 0.0, serial, "request", (neighbor, tuple(path + [neighbor]))),
                )
                serial += 1

        elif event_type == "response":
            candidate, path_tuple, trust_value = payload
            path = list(path_tuple)
            if candidate in returned_candidates:
                continue
            returned_candidates.add(candidate)

            if decision_time is not None or time_ms > cfg.deadline_ms:
                cached_after_decision += 1
                continue
            candidates_in_time += 1

            if not use_confirmation:
                valid = (
                    online_at(env, candidate, time_ms)
                    and path[-2] in env.actual_adj[candidate]
                    and qualifies(resource_at(env, candidate, time_ms), env.demand)
                )
                decision_time = time_ms
                selected_hops = len(path) - 1
                selected_trust = trust_value
                selected_candidate = int(candidate) if valid else -1
                success = valid
                if not valid:
                    stale_or_invalid += 1
                update_beta_after_confirmation(env, path, valid)
                continue

            if len(confirmation_candidates) >= cfg.top_k or candidate in confirmation_candidates:
                continue
            confirmation_candidates.add(candidate)
            confirmation_attempts += 1
            confirmation_messages += 2
            confirmation_time = direct_confirmation_arrival(env, candidate, time_ms, cfg, rng)
            heapq.heappush(
                events,
                (
                    confirmation_time,
                    2,
                    -trust_value,
                    serial,
                    "confirmation",
                    (candidate, tuple(path), trust_value),
                ),
            )
            serial += 1

        elif event_type == "confirmation":
            candidate, path_tuple, trust_value = payload
            path = list(path_tuple)
            if decision_time is not None:
                cached_after_decision += 1
                continue
            if time_ms > cfg.deadline_ms:
                cached_after_decision += 1
                continue
            valid = (
                online_at(env, candidate, time_ms)
                and path[-2] in env.actual_adj[candidate]
                and qualifies(resource_at(env, candidate, time_ms), env.demand)
            )
            update_beta_after_confirmation(env, path, valid)
            if valid:
                success = True
                decision_time = time_ms
                selected_hops = len(path) - 1
                selected_trust = trust_value
                selected_candidate = int(candidate)
            else:
                stale_or_invalid += 1

    latency = decision_time if decision_time is not None else cfg.deadline_ms
    return SearchResult(
        method=method,
        success=int(success),
        decision_latency_ms=float(latency),
        message_count=request_messages + response_messages + confirmation_messages,
        request_messages=request_messages,
        response_messages=response_messages,
        confirmation_messages=confirmation_messages,
        candidates_in_time=candidates_in_time,
        confirmation_attempts=confirmation_attempts,
        stale_or_invalid_candidates=stale_or_invalid,
        cached_after_decision=cached_after_decision,
        selected_hops=float(selected_hops),
        selected_trust=float(selected_trust),
        selected_candidate=int(selected_candidate),
    )


def run_paired_trials(
    cfg: SearchConfig,
    methods: Sequence[str],
    trials: int,
    seed_offset: int,
    scenario: str,
    scenario_value: float,
) -> pd.DataFrame:
    rows = []
    for trial in range(trials):
        env_seed = seed_offset + trial
        env = generate_environment(env_seed, cfg)
        for method_idx, method in enumerate(methods):
            result = run_search(method, env, cfg, seed=10_000_000 + env_seed * 17 + method_idx)
            row = asdict(result)
            row.update(
                {
                    "trial": trial,
                    "environment_seed": env_seed,
                    "scenario": scenario,
                    "scenario_value": scenario_value,
                    "num_nodes": cfg.num_nodes,
                    "average_degree": cfg.average_degree,
                    "max_hops": cfg.max_hops,
                    "deadline_ms": cfg.deadline_ms,
                    "outage_rate": cfg.outage_rate,
                    "topology_churn_rate": cfg.topology_churn_rate,
                    "success_latency_ms": result.decision_latency_ms if result.success else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    def q95(series: pd.Series) -> float:
        return float(series.quantile(0.95))

    grouped = raw.groupby(["scenario", "scenario_value", "method"], sort=False)
    summary = grouped.agg(
        trials=("success", "size"),
        success_rate=("success", "mean"),
        success_rate_std=("success", "std"),
        mean_latency_ms=("decision_latency_ms", "mean"),
        mean_success_latency_ms=("success_latency_ms", "mean"),
        p95_success_latency_ms=("success_latency_ms", q95),
        mean_messages=("message_count", "mean"),
        std_messages=("message_count", "std"),
        mean_candidates=("candidates_in_time", "mean"),
        mean_confirmations=("confirmation_attempts", "mean"),
        mean_invalid_candidates=("stale_or_invalid_candidates", "mean"),
        mean_cached_after_decision=("cached_after_decision", "mean"),
    ).reset_index()
    summary["success_rate_percent"] = 100.0 * summary["success_rate"]
    summary["success_ci95_percent"] = 100.0 * 1.96 * np.sqrt(
        summary["success_rate"] * (1.0 - summary["success_rate"]) / summary["trials"]
    )
    summary["invalid_per_confirmation"] = np.where(
        summary["mean_confirmations"] > 0,
        summary["mean_invalid_candidates"] / summary["mean_confirmations"],
        summary["mean_invalid_candidates"],
    )
    return summary


def _plot_method_lines(ax, data: pd.DataFrame, y: str, ylabel: str, scale_x: float = 1.0) -> None:
    for method in ["workflow_confirm", "workflow_no_confirm", "gossip_confirm"]:
        subset = data[data["method"] == method].sort_values("scenario_value")
        if subset.empty:
            continue
        x = subset["scenario_value"].to_numpy() * scale_x
        ax.plot(
            x,
            subset[y],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.7,
            markersize=5.0,
            label=METHOD_LABELS[method],
        )
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)

    outage = summary[summary["scenario"] == "outage_rate"]
    _plot_method_lines(axes[0, 0], outage, "success_rate_percent", "Discovery success (%)", 100.0)
    axes[0, 0].set_xlabel("Nodes experiencing an outage (%)")
    axes[0, 0].set_title("(a) Robustness to node outages", loc="left", fontweight="bold")
    axes[0, 0].set_ylim(0, 100)

    _plot_method_lines(axes[0, 1], outage, "mean_messages", "Messages per request", 100.0)
    axes[0, 1].set_xlabel("Nodes experiencing an outage (%)")
    axes[0, 1].set_title("(b) Communication overhead", loc="left", fontweight="bold")

    deadline = summary[
        (summary["scenario"] == "deadline_ms") & (summary["method"] == "workflow_confirm")
    ].sort_values("scenario_value")
    axes[1, 0].plot(
        deadline["scenario_value"], deadline["success_rate_percent"],
        color=COLORS["workflow_confirm"], marker="o", linewidth=1.7, markersize=5
    )
    axes[1, 0].set_xlabel("Absolute response deadline (ms)")
    axes[1, 0].set_ylabel("Discovery success (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1, 0].spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_title("(c) Deadline sensitivity", loc="left", fontweight="bold")

    depth = summary[
        (summary["scenario"] == "max_hops") & (summary["method"] == "workflow_confirm")
    ].sort_values("scenario_value")
    axes[1, 1].plot(
        depth["scenario_value"], depth["success_rate_percent"],
        color=COLORS["workflow_confirm"], marker="o", linewidth=1.7, markersize=5
    )
    axes[1, 1].set_xlabel("Maximum forwarding depth")
    axes[1, 1].set_ylabel("Discovery success (%)")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_xticks(depth["scenario_value"])
    axes[1, 1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    axes[1, 1].set_title("(d) Forwarding-depth sensitivity", loc="left", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=3, frameon=False)
    for suffix, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 300},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(output_dir / f"dynamic_search_evaluation.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    churn = summary[summary["scenario"] == "topology_churn_rate"]
    _plot_method_lines(axes[0], churn, "success_rate_percent", "Discovery success (%)", 100.0)
    axes[0].set_xlabel("Changed business links (%)")
    axes[0].set_title("(a) Dynamic business topology", loc="left", fontweight="bold")
    axes[0].set_ylim(0, 100)

    resource_change = summary[summary["scenario"] == "resource_change_rate"]
    _plot_method_lines(axes[1], resource_change, "success_rate_percent", "Discovery success (%)", 100.0)
    axes[1].set_xlabel("Nodes with an in-search availability drop (%)")
    axes[1].set_title("(b) Dynamic resource availability", loc="left", fontweight="bold")
    axes[1].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)
    for suffix, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 300},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(output_dir / f"dynamic_topology_resource_robustness.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def paired_success_tests(raw: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import binomtest

    default = raw[
        (raw["scenario"] == "outage_rate") & np.isclose(raw["scenario_value"], 0.10)
    ]
    rows = []
    for baseline in ["workflow_no_confirm", "gossip_confirm"]:
        proposed = default[default["method"] == "workflow_confirm"][
            ["environment_seed", "success"]
        ].rename(columns={"success": "proposed_success"})
        comparator = default[default["method"] == baseline][
            ["environment_seed", "success"]
        ].rename(columns={"success": "baseline_success"})
        paired = proposed.merge(comparator, on="environment_seed", validate="one_to_one")
        proposed_only = int(((paired["proposed_success"] == 1) & (paired["baseline_success"] == 0)).sum())
        baseline_only = int(((paired["proposed_success"] == 0) & (paired["baseline_success"] == 1)).sum())
        discordant = proposed_only + baseline_only
        p_value = (
            float(binomtest(proposed_only, discordant, 0.5, alternative="two-sided").pvalue)
            if discordant
            else 1.0
        )
        rows.append(
            {
                "comparison": f"workflow_confirm vs {baseline}",
                "trials": len(paired),
                "proposed_success_percent": 100.0 * paired["proposed_success"].mean(),
                "baseline_success_percent": 100.0 * paired["baseline_success"].mean(),
                "difference_percentage_points": 100.0
                * (paired["proposed_success"].mean() - paired["baseline_success"].mean()),
                "proposed_only_successes": proposed_only,
                "baseline_only_successes": baseline_only,
                "mcnemar_exact_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    paired_tests: pd.DataFrame,
    output_dir: Path,
    cfg: SearchConfig,
    trials: int,
) -> None:
    outage = summary[summary["scenario"] == "outage_rate"].copy()
    base = outage[np.isclose(outage["scenario_value"], 0.10)].copy()
    deadline = summary[
        (summary["scenario"] == "deadline_ms") & (summary["method"] == "workflow_confirm")
    ].sort_values("scenario_value")
    depth = summary[
        (summary["scenario"] == "max_hops") & (summary["method"] == "workflow_confirm")
    ].sort_values("scenario_value")
    churn = summary[summary["scenario"] == "topology_churn_rate"].copy()
    resource_change = summary[summary["scenario"] == "resource_change_rate"].copy()

    def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
        selected = frame.loc[:, columns]
        header = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join(["---"] * len(columns)) + "|"
        rows = []
        for values in selected.itertuples(index=False, name=None):
            formatted = []
            for value in values:
                if isinstance(value, (float, np.floating)):
                    formatted.append("NA" if np.isnan(value) else f"{value:.3f}")
                else:
                    formatted.append(str(value))
            rows.append("| " + " | ".join(formatted) + " |")
        return "\n".join([header, separator, *rows])

    report = f"""# 动态节点搜索统一实验结果与分析

## 1. 实验目的

本实验针对审稿人提出的静态拓扑、节点中途断联、最大响应时间、资源二次确认和Gossip对照问题。实验保留原论文沿业务关系逐跳搜索的核心逻辑，并采用离散事件仿真显式计算请求、响应和确认消息的网络时延。

## 2. 统一设置

- ECN数量：{cfg.num_nodes}
- 平均业务度数：{cfg.average_degree}
- 单跳单向时延：{cfg.one_way_delay_ms:.0f} ms，加入标准差{cfg.delay_jitter_ms:.0f} ms的链路抖动
- 默认最大转发深度：{cfg.max_hops}
- 默认绝对截止时间：{cfg.deadline_ms:.0f} ms
- 二次确认：对最先返回的top-{cfg.top_k}候选并行/异步确认资源、存活状态和当前业务关系
- Gossip基线：相同图、截止时间和确认机制，每个中间节点随机选择最多{cfg.gossip_fanout}个邻居继续传播
- 每个参数点：{trials}个配对随机环境；不同方法使用相同拓扑、资源、请求和故障
- 多跳搜索仅在直接业务邻居均不满足缓存资源门槛时触发
- 直接信任：Beta reputation，$T=(1+s)/(2+s+f)$；间接信任按论文的路径衰减公式递归传播
- 拓扑变化：节点仍持有变化前的短期邻居视图，实际链路已有部分删除/替换
- 延迟结果：只进入带时间戳的候选缓存，不参与已开始的卸载

## 3. 默认动态场景结果

默认场景包含10%节点发生中途断联、10%业务链路变化，以及30%节点在搜索期间因新任务到达而发生可用资源下降。

{markdown_table(base.assign(method=base['method'].map(METHOD_LABELS)), ['method', 'success_rate_percent', 'success_ci95_percent', 'mean_success_latency_ms', 'p95_success_latency_ms', 'mean_messages', 'invalid_per_confirmation', 'mean_cached_after_decision'])}

`Workflow + confirmation`是修订后的完整机制；`Workflow, no confirmation`用于量化二次确认的作用；`Gossip + confirmation`是固定fanout的去中心化查询传播基线。

配对成功率比较采用同一环境种子的精确McNemar检验：

{markdown_table(paired_tests, ['comparison', 'trials', 'proposed_success_percent', 'baseline_success_percent', 'difference_percentage_points', 'mcnemar_exact_p'])}

## 4. 节点断联鲁棒性

{markdown_table(outage.assign(method=outage['method'].map(METHOD_LABELS)), ['scenario_value', 'method', 'success_rate_percent', 'success_ci95_percent', 'p95_success_latency_ms', 'mean_messages'])}

## 5. 截止时间敏感性

{markdown_table(deadline, ['scenario_value', 'success_rate_percent', 'success_ci95_percent', 'p95_success_latency_ms', 'mean_messages'])}

截止时间过短会使有效候选的响应或二次确认无法在本轮搜索中完成。延迟返回的信息被缓存，但不会回溯修改已经开始的卸载。

## 6. 最大转发深度敏感性

{markdown_table(depth, ['scenario_value', 'success_rate_percent', 'success_ci95_percent', 'p95_success_latency_ms', 'mean_messages'])}

最大转发深度提高通常会增加成功率，但也会增加消息开销；当成功率趋于饱和后，应选择拐点附近的深度，而不是无约束增加Dmax。

## 7. 业务连接动态变化

{markdown_table(churn.assign(method=churn['method'].map(METHOD_LABELS)), ['scenario_value', 'method', 'success_rate_percent', 'success_ci95_percent', 'p95_success_latency_ms', 'mean_messages'])}

这里的变化率表示搜索节点持有的短期业务邻接视图与实际可用链路之间的差异比例。被删除的旧链路会造成请求或响应丢失；新建立但尚未同步的链路不会被本轮搜索使用。

## 8. 搜索期间可用资源下降

{markdown_table(resource_change.assign(method=resource_change['method'].map(METHOD_LABELS)), ['scenario_value', 'method', 'success_rate_percent', 'success_ci95_percent', 'invalid_per_confirmation', 'mean_messages'])}

该组实验是单侧压力测试：一定比例的节点在搜索过程中因新任务到达而降低可用CPU和内存。它用于量化缓存资源与确认时真实资源之间的偏差。带二次确认的方法会删除失活、资源不足或业务关系已经失效的候选；无确认方法可能直接开始一次不可执行的卸载。

## 9. 结果边界

1. 这是可重复的离散事件仿真，不是真实网络测试台；所有网络时延均来自显式配置，而不是Python代码执行时间。
2. Gossip对照是通用的固定fanout查询传播协议，不等同于论文参考文献[15]中的Gossip Learning预测训练协议。正文必须写作“gossip-style decentralized query dissemination baseline”。
3. 二次确认假设候选IP返回后可进行直接请求/响应；若实际工业网络禁止跨业务链路直连，需要改为沿返回路径确认。
4. Beta reputation使仿真自包含，但本文创新重点仍是路径信任传播，不应声称提出了新的直接信任估计方法。
"""
    (output_dir / "动态节点搜索统一实验结果与分析.md").write_text(report, encoding="utf-8")


def run_all(output_dir: Path, trials: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ["workflow_confirm", "workflow_no_confirm", "gossip_confirm"]
    raw_frames = []
    for outage_rate in [0.0, 0.05, 0.10, 0.20]:
        cfg = SearchConfig(outage_rate=outage_rate, topology_churn_rate=0.10)
        raw_frames.append(
            run_paired_trials(cfg, methods, trials, 1000, "outage_rate", outage_rate)
        )

    for deadline_ms in [60.0, 90.0, 120.0, 150.0, 180.0, 240.0]:
        cfg = SearchConfig(deadline_ms=deadline_ms, outage_rate=0.10, topology_churn_rate=0.10)
        raw_frames.append(
            run_paired_trials(cfg, ["workflow_confirm"], trials, 100_000, "deadline_ms", deadline_ms)
        )

    for churn_rate in [0.0, 0.05, 0.10, 0.20]:
        cfg = SearchConfig(outage_rate=0.10, topology_churn_rate=churn_rate)
        raw_frames.append(
            run_paired_trials(cfg, methods, trials, 300_000, "topology_churn_rate", churn_rate)
        )

    for resource_change_rate in [0.0, 0.10, 0.30, 0.50]:
        cfg = SearchConfig(
            outage_rate=0.10,
            topology_churn_rate=0.10,
            resource_change_probability=resource_change_rate,
        )
        raw_frames.append(
            run_paired_trials(
                cfg,
                ["workflow_confirm", "workflow_no_confirm"],
                trials,
                400_000,
                "resource_change_rate",
                resource_change_rate,
            )
        )

    for max_hops in [1, 2, 3, 4, 5, 6]:
        cfg = SearchConfig(max_hops=max_hops, outage_rate=0.10, topology_churn_rate=0.10)
        raw_frames.append(
            run_paired_trials(cfg, ["workflow_confirm"], trials, 200_000, "max_hops", float(max_hops))
        )

    raw = pd.concat(raw_frames, ignore_index=True)
    summary = summarize(raw)
    paired_tests = paired_success_tests(raw)
    raw.to_csv(output_dir / "dynamic_search_raw.csv", index=False)
    summary.to_csv(output_dir / "dynamic_search_summary.csv", index=False)
    paired_tests.to_csv(output_dir / "paired_success_tests.csv", index=False)
    cfg = SearchConfig()
    plot_results(summary, output_dir)
    write_report(summary, paired_tests, output_dir, cfg, trials)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "trials_per_parameter_point": trials,
        "default_config": asdict(cfg),
        "method_definitions": METHOD_LABELS,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic workflow-based node-search experiments")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "local_py310",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args.output_dir, args.trials)
