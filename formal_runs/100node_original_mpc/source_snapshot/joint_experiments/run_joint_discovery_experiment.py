from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent
sys.path.insert(0, str(RAFT_DIR / "update_experiments"))
sys.path.insert(0, str(RAFT_DIR / "search_experiments"))

from run_update_experiments import (  # noqa: E402
    BudgetGuard,
    load_inputs,
    mpc_dp_first_action,
    weighted_squared_error,
)
from run_dynamic_search_experiments import (  # noqa: E402
    Environment,
    SearchConfig,
    beta_reputation,
    churn_graph,
    create_connected_business_graph,
    run_search,
)


PIPELINES = {
    "proposed": "FITS + MPC-DP + Workflow",
    "last_mpc": "Last-value + MPC-DP + Workflow",
    "fits_periodic": "FITS + Periodic + Workflow",
    "fits_delta": "FITS + Send-on-delta + Workflow",
    "fits_gossip": "FITS + MPC-DP + Gossip",
    "oracle_state": "Fresh-state oracle + Workflow",
}

COLORS = {
    "proposed": "#0072B2",
    "last_mpc": "#CC79A7",
    "fits_periodic": "#E69F00",
    "fits_delta": "#D55E00",
    "fits_gossip": "#009E73",
    "oracle_state": "#666666",
}


@dataclass(frozen=True)
class JointConfig:
    num_nodes: int = 10
    trace_length: int = 300
    trace_start: int = 94
    offset_step: int = 8
    average_degree: int = 3
    update_budget: float = 0.5
    fits_eta: float = 1.0
    last_eta: float = 2.0
    delta_threshold: float = 1.3355422455022612
    horizon: int = 6
    tasks_per_network: int = 120
    network_repeats: int = 40
    outage_rate: float = 0.05
    topology_churn_rate: float = 0.05
    max_hops: int = 5
    deadline_ms: float = 180.0
    one_way_delay_ms: float = 15.0
    top_k: int = 3
    gossip_fanout: int = 2
    resource_change_probability: float = 0.0
    resource_shock_std: float = 0.0


@dataclass
class CacheReplay:
    pipeline: str
    cache_utilization: np.ndarray
    actions: np.ndarray
    cache_wrmse: float
    update_count: int
    update_rate: float


def build_node_replays(
    actual: np.ndarray, predictions: np.ndarray, cfg: JointConfig
) -> Tuple[np.ndarray, np.ndarray]:
    nodes_actual = []
    nodes_predictions = []
    for node in range(cfg.num_nodes):
        start = cfg.trace_start + node * cfg.offset_step
        stop = start + cfg.trace_length
        if stop > len(actual):
            raise ValueError(f"Trace too short for node {node}: need {stop}, have {len(actual)}")
        nodes_actual.append(actual[start:stop])
        nodes_predictions.append(predictions[start:stop])
    return np.asarray(nodes_actual), np.asarray(nodes_predictions)


def last_value_forecast(node_actual: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(node_actual[:, None, :], horizon, axis=1)


def replay_one_node(
    actual: np.ndarray,
    predictions: np.ndarray,
    strategy: str,
    cfg: JointConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    length = len(actual)
    cache = actual[0].copy()
    cache_history = np.empty_like(actual)
    actions = np.zeros(length, dtype=bool)
    guard = BudgetGuard(length, cfg.update_budget)

    for t in range(length):
        if strategy == "fits_mpc":
            forecast = predictions[t, : cfg.horizon]
            local_budget = min(
                guard.target - guard.used,
                int(math.ceil(cfg.update_budget * len(forecast))),
            )
            proposed = mpc_dp_first_action(forecast, cache, local_budget, cfg.fits_eta)
        elif strategy == "last_mpc":
            forecast = predictions[t, : cfg.horizon]
            local_budget = min(
                guard.target - guard.used,
                int(math.ceil(cfg.update_budget * len(forecast))),
            )
            proposed = mpc_dp_first_action(forecast, cache, local_budget, cfg.last_eta)
        elif strategy == "periodic":
            before = int(math.floor(t * guard.target / length))
            after = int(math.floor((t + 1) * guard.target / length))
            proposed = after > before
        elif strategy == "delta":
            delta = math.sqrt(weighted_squared_error(actual[t], cache))
            proposed = delta > cfg.delta_threshold
        elif strategy == "oracle":
            cache_history[t] = actual[t]
            actions[t] = True
            continue
        else:
            raise ValueError(strategy)

        action = guard.apply(proposed, t)
        actions[t] = action
        if action:
            cache = actual[t].copy()
        cache_history[t] = cache

    if strategy != "oracle" and int(actions.sum()) != guard.target:
        raise AssertionError(f"{strategy}: {actions.sum()} != {guard.target}")
    return cache_history, actions


def build_cache_replays(
    nodes_actual: np.ndarray, nodes_predictions: np.ndarray, cfg: JointConfig
) -> Dict[str, CacheReplay]:
    strategy_for_pipeline = {
        "proposed": "fits_mpc",
        "last_mpc": "last_mpc",
        "fits_periodic": "periodic",
        "fits_delta": "delta",
        "fits_gossip": "fits_mpc",
        "oracle_state": "oracle",
    }
    replays: Dict[str, CacheReplay] = {}
    shared: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for pipeline, strategy in strategy_for_pipeline.items():
        if strategy not in shared:
            caches = []
            actions = []
            for node in range(cfg.num_nodes):
                prediction = (
                    last_value_forecast(nodes_actual[node], cfg.horizon)
                    if strategy == "last_mpc"
                    else nodes_predictions[node]
                )
                cache, action = replay_one_node(nodes_actual[node], prediction, strategy, cfg)
                caches.append(cache)
                actions.append(action)
            shared[strategy] = (np.asarray(caches), np.asarray(actions))
        cache_array, action_array = shared[strategy]
        errors = nodes_actual - cache_array
        wrmse = float(np.sqrt(np.mean(np.sum(0.5 * np.square(errors), axis=2))))
        replays[pipeline] = CacheReplay(
            pipeline=pipeline,
            cache_utilization=cache_array,
            actions=action_array,
            cache_wrmse=wrmse,
            update_count=int(action_array.sum()),
            update_rate=float(action_array.mean()),
        )
    return replays


def generate_tasks(
    nodes_actual: np.ndarray,
    cfg: JointConfig,
    seed: int,
    graph: Sequence[Set[int]] | None = None,
    remote_required: bool = False,
) -> List[Tuple[int, int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(cfg.tasks_per_network):
        for _attempt in range(100):
            time_slot = int(rng.integers(10, cfg.trace_length))
            origin = int(rng.integers(0, cfg.num_nodes))
            availability = 100.0 - nodes_actual[:, time_slot]
            eligible_targets = [
                node
                for node in range(cfg.num_nodes)
                if node != origin
                and (not remote_required or (graph is not None and node not in graph[origin]))
                and availability[node, 0] >= 30.0
                and availability[node, 1] >= 4.0
            ]
            if not eligible_targets:
                continue
            target = int(rng.choice(eligible_targets))
            closeness = rng.uniform(0.90, 0.99, size=2)
            demand = np.maximum(availability[target] * closeness, np.array([5.0, 1.0]))
            if remote_required and graph is not None:
                if any(np.all(availability[node] >= demand) for node in graph[origin]):
                    continue
            tasks.append((time_slot, origin, demand))
            break
        else:
            raise RuntimeError("Could not generate a feasible task")
    return tasks


def graph_and_trust(cfg: JointConfig, seed: int):
    ss = np.random.SeedSequence(seed)
    rng_graph, rng_trust = [np.random.default_rng(x) for x in ss.spawn(2)]
    graph = create_connected_business_graph(cfg.num_nodes, cfg.average_degree, rng_graph)
    alpha, beta = beta_reputation(graph, rng_trust)
    return graph, alpha, beta


def environment_for_task(
    graph: Sequence[Set[int]],
    alpha,
    beta,
    actual_availability: np.ndarray,
    cached_availability: np.ndarray,
    origin: int,
    demand: np.ndarray,
    cfg: JointConfig,
    seed: int,
) -> Environment:
    ss = np.random.SeedSequence(seed)
    rng_churn, rng_outage, rng_resource = [np.random.default_rng(x) for x in ss.spawn(3)]
    actual_graph = churn_graph(graph, cfg.topology_churn_rate, rng_churn)
    outage_start = np.full(cfg.num_nodes, np.inf)
    outage_end = np.full(cfg.num_nodes, np.inf)
    mask = rng_outage.random(cfg.num_nodes) < cfg.outage_rate
    outage_start[mask] = rng_outage.uniform(0.0, 90.0, int(mask.sum()))
    outage_end[mask] = outage_start[mask] + rng_outage.uniform(60.0, 180.0, int(mask.sum()))
    outage_start[origin] = np.inf
    outage_end[origin] = np.inf
    resource_pre = actual_availability.copy()
    resource_post = actual_availability.copy()
    resource_change_time = np.full(cfg.num_nodes, np.inf)
    if cfg.resource_change_probability > 0.0 and cfg.resource_shock_std > 0.0:
        changed = rng_resource.random(cfg.num_nodes) < cfg.resource_change_probability
        changed[origin] = False
        resource_post[changed] = np.clip(
            resource_pre[changed]
            - np.abs(
                rng_resource.normal(
                    0.0,
                    cfg.resource_shock_std,
                    size=(int(changed.sum()), resource_pre.shape[1]),
                )
            ),
            0.0,
            100.0,
        )
        resource_change_time[changed] = rng_resource.uniform(
            0.15 * cfg.deadline_ms,
            0.65 * cfg.deadline_ms,
            int(changed.sum()),
        )
    return Environment(
        known_adj=[set(x) for x in graph],
        actual_adj=actual_graph,
        cache_resource=cached_availability.copy(),
        resource_pre=resource_pre,
        resource_post=resource_post,
        resource_change_time=resource_change_time,
        outage_start=outage_start,
        outage_end=outage_end,
        trust_alpha=copy.deepcopy(alpha),
        trust_beta=copy.deepcopy(beta),
        origin=origin,
        demand=demand.copy(),
    )


def run_joint_trials(
    nodes_actual: np.ndarray,
    replays: Dict[str, CacheReplay],
    cfg: JointConfig,
    pipelines: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows = []
    selected_pipelines = list(pipelines) if pipelines is not None else list(replays)
    unknown = set(selected_pipelines) - set(replays)
    if unknown:
        raise ValueError(f"Unknown pipelines: {sorted(unknown)}")
    search_cfg = SearchConfig(
        num_nodes=cfg.num_nodes,
        average_degree=cfg.average_degree,
        max_hops=cfg.max_hops,
        deadline_ms=cfg.deadline_ms,
        one_way_delay_ms=cfg.one_way_delay_ms,
        top_k=cfg.top_k,
        gossip_fanout=cfg.gossip_fanout,
        outage_rate=cfg.outage_rate,
        topology_churn_rate=cfg.topology_churn_rate,
        resource_change_probability=0.0,
    )
    for repeat in range(cfg.network_repeats):
        graph_seed = 50_000 + repeat
        graph, alpha, beta = graph_and_trust(cfg, graph_seed)
        degree_sum = sum(len(x) for x in graph)
        workloads = {
            "general_feasible": generate_tasks(
                nodes_actual, cfg, seed=70_000 + repeat, graph=graph, remote_required=False
            ),
            "remote_required": generate_tasks(
                nodes_actual, cfg, seed=170_000 + repeat, graph=graph, remote_required=True
            ),
        }
        for workload_idx, (workload, tasks) in enumerate(workloads.items()):
            for task_idx, (time_slot, origin, demand) in enumerate(tasks):
                actual_availability = 100.0 - nodes_actual[:, time_slot]
                for pipeline in selected_pipelines:
                    replay = replays[pipeline]
                    cached_availability = 100.0 - replay.cache_utilization[:, time_slot]
                    true_eligible = np.all(actual_availability >= demand, axis=1)
                    cached_eligible = np.all(cached_availability >= demand, axis=1)
                    true_eligible[origin] = False
                    cached_eligible[origin] = False
                    true_positive = int(np.sum(true_eligible & cached_eligible))
                    true_count = int(np.sum(true_eligible))
                    cached_count = int(np.sum(cached_eligible))
                    false_negative = int(np.sum(true_eligible & ~cached_eligible))
                    false_positive = int(np.sum(~true_eligible & cached_eligible))
                    candidate_recall = true_positive / true_count if true_count else 1.0
                    candidate_precision = true_positive / cached_count if cached_count else np.nan
                    env = environment_for_task(
                        graph,
                        alpha,
                        beta,
                        actual_availability,
                        cached_availability,
                        origin,
                        demand,
                        cfg,
                        seed=1_000_000
                        + workload_idx * 500_000
                        + repeat * cfg.tasks_per_network
                        + task_idx,
                    )
                    search_method = "gossip_confirm" if pipeline == "fits_gossip" else "workflow_confirm"
                    result = run_search(
                        search_method,
                        env,
                        search_cfg,
                        seed=2_000_000
                        + workload_idx * 500_000
                        + repeat * cfg.tasks_per_network
                        + task_idx,
                    )
                    update_message_total = int(
                        sum(replay.actions[node].sum() * len(graph[node]) for node in range(cfg.num_nodes))
                    )
                    amortized_update_messages = update_message_total / cfg.tasks_per_network
                    rows.append(
                        {
                            "repeat": repeat,
                            "workload": workload,
                            "task": task_idx,
                            "task_key": workload_idx * 10_000_000 + repeat * cfg.tasks_per_network + task_idx,
                            "pipeline": pipeline,
                            "time_slot": time_slot,
                            "origin": origin,
                            "success": result.success,
                            "success_latency_ms": result.decision_latency_ms if result.success else np.nan,
                            "censored_latency_ms": result.decision_latency_ms,
                            "search_messages": result.message_count,
                            "amortized_update_messages": amortized_update_messages,
                            "amortized_total_messages": result.message_count + amortized_update_messages,
                            "stale_or_invalid_candidates": result.stale_or_invalid_candidates,
                            "confirmation_attempts": result.confirmation_attempts,
                            "selected_hops": result.selected_hops,
                            "direct_success": int(result.success and np.isclose(result.selected_hops, 1.0)),
                            "multihop_success": int(result.success and result.selected_hops > 1.0),
                            "cache_wrmse": replay.cache_wrmse,
                            "candidate_recall": candidate_recall,
                            "candidate_precision": candidate_precision,
                            "true_candidate_count": true_count,
                            "cached_candidate_count": cached_count,
                            "true_positive_candidates": true_positive,
                            "false_negative_candidates": false_negative,
                            "false_positive_candidates": false_positive,
                            "cache_has_true_candidate": int(true_positive > 0),
                            "cache_blocked_discovery": int(true_positive == 0),
                            "protocol_failure_with_candidate": int(true_positive > 0 and not result.success),
                            "selected_trust": result.selected_trust if result.success else np.nan,
                            "update_rate": replay.update_rate,
                            "degree_sum": degree_sum,
                        }
                    )
    return pd.DataFrame(rows)


def percentile95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.groupby(["workload", "pipeline"], sort=False).agg(
        tasks=("success", "size"),
        discovery_success=("success", "mean"),
        mean_success_latency_ms=("success_latency_ms", "mean"),
        p95_success_latency_ms=("success_latency_ms", percentile95),
        mean_search_messages=("search_messages", "mean"),
        mean_update_messages=("amortized_update_messages", "mean"),
        mean_total_messages=("amortized_total_messages", "mean"),
        mean_invalid_candidates=("stale_or_invalid_candidates", "mean"),
        mean_confirmations=("confirmation_attempts", "mean"),
        direct_success_rate=("direct_success", "mean"),
        multihop_success_rate=("multihop_success", "mean"),
        cache_wrmse=("cache_wrmse", "mean"),
        candidate_recall=("candidate_recall", "mean"),
        candidate_precision=("candidate_precision", "mean"),
        true_candidate_count=("true_candidate_count", "sum"),
        cached_candidate_count=("cached_candidate_count", "sum"),
        true_positive_candidates=("true_positive_candidates", "sum"),
        false_negative_candidates=("false_negative_candidates", "sum"),
        false_positive_candidates=("false_positive_candidates", "sum"),
        cache_has_true_candidate_rate=("cache_has_true_candidate", "mean"),
        cache_blocked_discovery_rate=("cache_blocked_discovery", "mean"),
        protocol_failure_with_candidate_rate=("protocol_failure_with_candidate", "mean"),
        mean_selected_trust=("selected_trust", "mean"),
        update_rate=("update_rate", "mean"),
    ).reset_index()
    frame["success_percent"] = 100.0 * frame.discovery_success
    frame["success_ci95_percent"] = 100.0 * 1.96 * np.sqrt(
        frame.discovery_success * (1.0 - frame.discovery_success) / frame.tasks
    )
    frame["invalid_per_confirmation"] = frame.mean_invalid_candidates / frame.mean_confirmations
    frame["direct_success_percent"] = 100.0 * frame.direct_success_rate
    frame["multihop_success_percent"] = 100.0 * frame.multihop_success_rate
    frame["candidate_micro_recall"] = (
        frame.true_positive_candidates / frame.true_candidate_count
    )
    frame["candidate_micro_precision"] = (
        frame.true_positive_candidates / frame.cached_candidate_count
    )
    frame["candidate_false_negative_rate"] = (
        frame.false_negative_candidates / frame.true_candidate_count
    )
    frame["candidate_false_discovery_rate"] = (
        frame.false_positive_candidates / frame.cached_candidate_count
    )
    frame["cache_blocked_discovery_percent"] = 100.0 * frame.cache_blocked_discovery_rate
    frame["protocol_failure_with_candidate_percent"] = (
        100.0 * frame.protocol_failure_with_candidate_rate
    )
    frame["conditional_protocol_success_percent"] = 100.0 * (
        frame.discovery_success / frame.cache_has_true_candidate_rate
    )
    return frame


def paired_tests(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for workload in raw.workload.unique():
        subset = raw[raw.workload == workload]
        proposed = subset[subset.pipeline == "proposed"][["task_key", "success"]].rename(
            columns={"success": "proposed"}
        )
        for comparator in ["last_mpc", "fits_periodic", "fits_delta", "fits_gossip"]:
            baseline = subset[subset.pipeline == comparator][["task_key", "success"]].rename(
                columns={"success": "baseline"}
            )
            pair = proposed.merge(baseline, on="task_key", validate="one_to_one")
            a = int(((pair.proposed == 1) & (pair.baseline == 0)).sum())
            b = int(((pair.proposed == 0) & (pair.baseline == 1)).sum())
            p = float(binomtest(a, a + b, 0.5).pvalue) if a + b else 1.0
            rows.append(
                {
                    "workload": workload,
                    "comparison": f"proposed vs {comparator}",
                    "tasks": len(pair),
                    "proposed_success_percent": 100.0 * pair.proposed.mean(),
                    "baseline_success_percent": 100.0 * pair.baseline.mean(),
                    "difference_percentage_points": 100.0 * (pair.proposed.mean() - pair.baseline.mean()),
                    "proposed_only_successes": a,
                    "baseline_only_successes": b,
                    "mcnemar_exact_p": p,
                }
            )
    return pd.DataFrame(rows)


def run_top_k_sensitivity(
    nodes_actual: np.ndarray,
    replays: Dict[str, CacheReplay],
    cfg: JointConfig,
    top_k_values: Sequence[int] = (1, 2, 3, 5),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate confirmation depth without changing tasks, states, or network samples."""
    raw_frames = []
    for top_k in top_k_values:
        trial = run_joint_trials(
            nodes_actual,
            replays,
            replace(cfg, top_k=int(top_k)),
            pipelines=["proposed"],
        )
        trial["top_k"] = int(top_k)
        raw_frames.append(trial)
    raw = pd.concat(raw_frames, ignore_index=True)
    summary = raw.groupby(["workload", "top_k"], sort=False).agg(
        tasks=("success", "size"),
        discovery_success=("success", "mean"),
        mean_success_latency_ms=("success_latency_ms", "mean"),
        p95_success_latency_ms=("success_latency_ms", percentile95),
        mean_confirmations=("confirmation_attempts", "mean"),
        mean_invalid_candidates=("stale_or_invalid_candidates", "mean"),
        mean_search_messages=("search_messages", "mean"),
        cache_blocked_discovery_rate=("cache_blocked_discovery", "mean"),
        protocol_failure_with_candidate_rate=("protocol_failure_with_candidate", "mean"),
    ).reset_index()
    summary["success_percent"] = 100.0 * summary.discovery_success
    summary["success_ci95_percent"] = 100.0 * 1.96 * np.sqrt(
        summary.discovery_success * (1.0 - summary.discovery_success) / summary.tasks
    )

    test_rows = []
    for workload in raw.workload.unique():
        subset = raw[raw.workload == workload]
        reference = subset[subset.top_k == 3][["task_key", "success"]].rename(
            columns={"success": "top3"}
        )
        for top_k in top_k_values:
            if top_k == 3:
                continue
            comparator = subset[subset.top_k == top_k][["task_key", "success"]].rename(
                columns={"success": "comparator"}
            )
            pair = reference.merge(comparator, on="task_key", validate="one_to_one")
            a = int(((pair.top3 == 1) & (pair.comparator == 0)).sum())
            b = int(((pair.top3 == 0) & (pair.comparator == 1)).sum())
            p_value = float(binomtest(a, a + b, 0.5).pvalue) if a + b else 1.0
            test_rows.append(
                {
                    "workload": workload,
                    "comparison": f"top-3 vs top-{top_k}",
                    "tasks": len(pair),
                    "top3_success_percent": 100.0 * pair.top3.mean(),
                    "comparator_success_percent": 100.0 * pair.comparator.mean(),
                    "difference_percentage_points": 100.0 * (pair.top3.mean() - pair.comparator.mean()),
                    "mcnemar_exact_p": p_value,
                }
            )
    return raw, summary, pd.DataFrame(test_rows)


def export_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        output_dir / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    order = ["proposed", "last_mpc", "fits_periodic", "fits_delta", "fits_gossip", "oracle_state"]
    general = summary[summary.workload == "general_feasible"].set_index("pipeline").loc[order].reset_index()
    remote = summary[summary.workload == "remote_required"].set_index("pipeline").loc[order].reset_index()
    labels = [
        "Prop.",
        "Last",
        "Periodic",
        "Delta",
        "Gossip",
        "Oracle",
    ]
    colors = [COLORS[x] for x in order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65), constrained_layout=True)

    axes[0, 0].bar(x, general.success_percent, color=colors, edgecolor="black", linewidth=0.4)
    axes[0, 0].errorbar(
        x,
        general.success_percent,
        yerr=general.success_ci95_percent,
        fmt="none",
        color="black",
        capsize=2,
        linewidth=0.7,
    )
    axes[0, 0].set_ylabel("Usable-candidate discovery (%)")
    axes[0, 0].set_title("(a) General feasible tasks", loc="left", fontweight="bold")
    axes[0, 0].set_ylim(0, 100)

    axes[0, 1].bar(x, remote.success_percent, color=colors, edgecolor="black", linewidth=0.4)
    axes[0, 1].errorbar(
        x,
        remote.success_percent,
        yerr=remote.success_ci95_percent,
        fmt="none",
        color="black",
        capsize=2,
        linewidth=0.7,
    )
    axes[0, 1].set_ylabel("Usable-candidate discovery (%)")
    axes[0, 1].set_title("(b) Remote-required tasks", loc="left", fontweight="bold")
    axes[0, 1].set_ylim(0, 100)

    axes[0, 2].bar(x, general.cache_wrmse, color=colors, edgecolor="black", linewidth=0.4)
    axes[0, 2].set_ylabel("State-cache WRMSE")
    axes[0, 2].set_title("(c) Cached-state accuracy", loc="left", fontweight="bold")

    axes[1, 0].bar(x, remote.candidate_recall * 100.0, color=colors, edgecolor="black", linewidth=0.4)
    axes[1, 0].set_ylabel("Cached-candidate recall (%)")
    axes[1, 0].set_title("(d) Threshold-level recall", loc="left", fontweight="bold")
    axes[1, 0].set_ylim(0, 100)

    axes[1, 1].bar(x, remote.mean_search_messages, color=colors, edgecolor="black", linewidth=0.4)
    axes[1, 1].set_ylabel("Search messages per task")
    axes[1, 1].set_title("(e) Remote-search overhead", loc="left", fontweight="bold")

    axes[1, 2].bar(x, remote.p95_success_latency_ms, color=colors, edgecolor="black", linewidth=0.4)
    axes[1, 2].set_ylabel("P95 successful-search latency (ms)")
    axes[1, 2].set_title("(f) Remote-search latency", loc="left", fontweight="bold")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    export_figure(fig, output_dir, "joint_discovery_evaluation")
    plt.close(fig)


def plot_candidate_diagnostics(
    summary: pd.DataFrame, top_k_summary: pd.DataFrame, output_dir: Path
) -> None:
    order = ["proposed", "last_mpc", "fits_periodic", "fits_delta", "fits_gossip", "oracle_state"]
    labels = ["Prop.", "Last", "Periodic", "Delta", "Gossip", "Oracle"]
    colors = [COLORS[x] for x in order]
    remote = summary[summary.workload == "remote_required"].set_index("pipeline").loc[order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), constrained_layout=True)

    axes[0, 0].bar(
        x,
        100.0 * remote.candidate_micro_recall,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0, 0].set_ylabel("Resource-candidate recall (%)")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title("(a) Avoiding false exclusion", loc="left", fontweight="bold")

    axes[0, 1].bar(
        x,
        100.0 * remote.candidate_micro_precision,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0, 1].set_ylabel("Resource-candidate precision (%)")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_title("(b) Avoiding stale inclusion", loc="left", fontweight="bold")

    width = 0.72
    axes[1, 0].bar(
        x,
        remote.cache_blocked_discovery_percent,
        width,
        color="#999999",
        label="No true candidate in cache",
    )
    axes[1, 0].bar(
        x,
        remote.protocol_failure_with_candidate_percent,
        width,
        bottom=remote.cache_blocked_discovery_percent,
        color="#E69F00",
        label="Search/confirmation failure",
    )
    axes[1, 0].set_ylabel("Remote discovery failure (%)")
    axes[1, 0].set_title("(c) Failure attribution", loc="left", fontweight="bold")
    axes[1, 0].legend(
        frameon=False,
        fontsize=6.2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
    )

    top_colors = {"general_feasible": "#0072B2", "remote_required": "#D55E00"}
    top_labels = {"general_feasible": "General", "remote_required": "Remote-required"}
    for workload in ["general_feasible", "remote_required"]:
        data = top_k_summary[top_k_summary.workload == workload].sort_values("top_k")
        axes[1, 1].errorbar(
            data.top_k,
            data.success_percent,
            yerr=data.success_ci95_percent,
            marker="o",
            markersize=4,
            linewidth=1.2,
            capsize=2,
            color=top_colors[workload],
            label=top_labels[workload],
        )
    axes[1, 1].set_xlabel("Maximum confirmation candidates, k")
    axes[1, 1].set_ylabel("Usable-candidate discovery (%)")
    axes[1, 1].set_xticks(sorted(top_k_summary.top_k.unique()))
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_title("(d) Top-k confirmation sensitivity", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=6.5)

    for ax in axes.flat[:3]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    for ax in axes.ravel():
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    export_figure(fig, output_dir, "candidate_discovery_diagnostics")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    values.append("NA")
                elif value != 0 and abs(value) < 1e-4:
                    values.append(f"{value:.3e}")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    top_k_summary: pd.DataFrame,
    top_k_tests: pd.DataFrame,
    cfg: JointConfig,
    output_dir: Path,
) -> None:
    display = summary.copy()
    display["pipeline"] = display.pipeline.map(PIPELINES)
    indexed = summary.set_index(["workload", "pipeline"])
    prop_general = indexed.loc[("general_feasible", "proposed")]
    prop_remote = indexed.loc[("remote_required", "proposed")]
    periodic_general = indexed.loc[("general_feasible", "fits_periodic")]
    periodic_remote = indexed.loc[("remote_required", "fits_periodic")]
    gossip_remote = indexed.loc[("remote_required", "fits_gossip")]
    top_k_remote = top_k_summary[top_k_summary.workload == "remote_required"].set_index("top_k")
    report = f"""# FITS–MPC-DP–动态节点搜索联合实验

## 1. 联合流程

Alibaba真实CPU/内存利用率轨迹及FITS六步预测 → 每个节点按更新策略发送状态 → 邻居形成缓存资源视图 → 可执行任务需求到达 → 直接邻居/多跳业务关系搜索 → Beta路径信任 → top-{cfg.top_k}存活与资源确认 → 返回可用候选。

## 2. 实验设置

- 节点数：{cfg.num_nodes}；平均业务度数：{cfg.average_degree}。
- 轨迹长度：每节点{cfg.trace_length}个时隙；通过同一Alibaba真实轨迹的不同时间偏移构造10个节点回放。
- 公平更新预算：除Fresh-state oracle外均为{cfg.update_budget:.0%}；FITS-MPC eta={cfg.fits_eta:g}，Last-value-MPC eta={cfg.last_eta:g}，均沿用验证集选择结果。
- 网络重复：{cfg.network_repeats}；每类任务每个网络{cfg.tasks_per_network}个，共{2 * cfg.network_repeats * cfg.tasks_per_network}个任务。一般可行任务允许直接邻居满足需求；远端必需任务保证真实直接邻居均无法满足，但至少存在一个真实远端可行节点。
- 动态条件：{cfg.outage_rate:.0%}节点中途断联、{cfg.topology_churn_rate:.0%}业务链路变化；单跳时延{cfg.one_way_delay_ms:.0f} ms；截止时间{cfg.deadline_ms:.0f} ms；Dmax={cfg.max_hops}。
- 任务需求以真实轨迹对应时隙的可用CPU/内存构造，并保证网络中至少存在一个真实可行节点。

## 3. 联合结果

{markdown_table(display, ['workload', 'pipeline', 'success_percent', 'success_ci95_percent', 'direct_success_percent', 'multihop_success_percent', 'cache_wrmse', 'candidate_micro_recall', 'candidate_micro_precision', 'candidate_false_negative_rate', 'candidate_false_discovery_rate', 'cache_blocked_discovery_percent', 'protocol_failure_with_candidate_percent', 'conditional_protocol_success_percent', 'p95_success_latency_ms', 'mean_search_messages', 'mean_update_messages', 'mean_total_messages', 'invalid_per_confirmation', 'mean_selected_trust'])}

## 4. 配对显著性检验

{markdown_table(tests, ['workload', 'comparison', 'tasks', 'proposed_success_percent', 'baseline_success_percent', 'difference_percentage_points', 'mcnemar_exact_p'])}

## 5. Top-k二次确认灵敏度

{markdown_table(top_k_summary, ['workload', 'top_k', 'tasks', 'success_percent', 'success_ci95_percent', 'p95_success_latency_ms', 'mean_confirmations', 'mean_invalid_candidates', 'mean_search_messages'])}

配对检验：

{markdown_table(top_k_tests, ['workload', 'comparison', 'tasks', 'top3_success_percent', 'comparator_success_percent', 'difference_percentage_points', 'mcnemar_exact_p'])}

## 6. 关键结果解释

- Proposed获得最低的公平预算缓存WRMSE（{prop_general.cache_wrmse:.4f}），但一般任务成功率为{prop_general.success_percent:.2f}%，低于Periodic的{periodic_general.success_percent:.2f}%。
- 在远端必需任务中，Proposed成功率为{prop_remote.success_percent:.2f}%，Periodic为{periodic_remote.success_percent:.2f}%，Gossip为{gossip_remote.success_percent:.2f}%。因此Proposed相对Gossip保留搜索覆盖优势，但不能声称联合性能优于Periodic。
- 原因是门槛分类与连续误差目标不一致：远端任务中Proposed的资源候选微平均召回率为{100 * prop_remote.candidate_micro_recall:.2f}%，Periodic为{100 * periodic_remote.candidate_micro_recall:.2f}%；对应的任务级缓存阻断率分别为{prop_remote.cache_blocked_discovery_percent:.2f}%和{periodic_remote.cache_blocked_discovery_percent:.2f}%。Periodic虽然总体数值误差更大，但较少把真实可用节点错误排除在资源门槛之外。
- 在缓存中至少保留一个真实可用节点的条件下，Proposed的搜索与确认成功率为{prop_remote.conditional_protocol_success_percent:.2f}%，因此最终失败既包含缓存假阴性，也包含动态拓扑、断联和截止时间造成的协议失败。
- Top-k灵敏度用于检验二次确认是否必要：远端任务的top-1、top-3和top-5发现成功率分别为{top_k_remote.loc[1, 'success_percent']:.2f}%、{top_k_remote.loc[3, 'success_percent']:.2f}%和{top_k_remote.loc[5, 'success_percent']:.2f}%。该结果用于选择确认深度，不能据此修改测试任务或资源门槛。
- 该结果说明当前MPC-DP适合优化连续状态感知误差，但尚未直接优化任务驱动的候选发现成功率。如果把联合发现作为论文核心结果，需要进一步设计门槛感知或任务需求感知的MPC代价函数，并在独立测试任务上验证。

## 7. 解释边界

1. 这是三模块联合轨迹回放，首次直接度量缓存质量如何影响最终可用候选发现，而不是完整任务执行/完成时间实验。
2. 10个ECN来自同一真实轨迹的错位回放，因此保留真实时间变化形态，但不能等同于10台独立工业设备测量。
3. Fresh-state oracle每时隙都拥有真实状态，是100%更新成本的不可部署上界，不属于相同通信预算的公平基线。
4. FITS只影响MPC-DP的更新时刻；Periodic和Send-on-delta不使用预测，因此名称中的FITS仅表示其余系统组件保持一致，不应暗示这些更新规则使用了FITS预测。
5. 结果仍是离散事件仿真，不是物理网络部署。
6. WRMSE衡量连续状态偏差，而候选发现依赖资源门槛两侧的分类。二者不必保持相同排序，因此报告同时给出缓存候选精确率和召回率。
"""
    (output_dir / "联合候选发现实验结果与分析.md").write_text(report, encoding="utf-8")


def run_all(data_dir: Path, output_dir: Path, cfg: JointConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    actual, predictions = load_inputs(data_dir)
    nodes_actual, nodes_predictions = build_node_replays(actual, predictions, cfg)
    replays = build_cache_replays(nodes_actual, nodes_predictions, cfg)
    raw = run_joint_trials(nodes_actual, replays, cfg)
    summary = summarize(raw)
    tests = paired_tests(raw)
    top_k_raw, top_k_summary, top_k_tests = run_top_k_sensitivity(
        nodes_actual, replays, cfg
    )
    raw.to_csv(output_dir / "joint_discovery_raw.csv", index=False)
    summary.to_csv(output_dir / "joint_discovery_summary.csv", index=False)
    tests.to_csv(output_dir / "joint_paired_tests.csv", index=False)
    top_k_raw.to_csv(output_dir / "top_k_sensitivity_raw.csv", index=False)
    top_k_summary.to_csv(output_dir / "top_k_sensitivity_summary.csv", index=False)
    top_k_tests.to_csv(output_dir / "top_k_sensitivity_tests.csv", index=False)
    plot_results(summary, output_dir)
    plot_candidate_diagnostics(summary, top_k_summary, output_dir)
    write_report(summary, tests, top_k_summary, top_k_tests, cfg, output_dir)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": asdict(cfg),
        "pipelines": PIPELINES,
        "data_dir": str(data_dir.resolve()),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nPaired tests\n", tests.to_string(index=False))
    print("\nTop-k sensitivity\n", top_k_summary.to_string(index=False))
    print("\nTop-k paired tests\n", top_k_tests.to_string(index=False))
    print(f"\nResults saved to {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "local_py310")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--tasks-per-network", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = JointConfig(network_repeats=args.repeats, tasks_per_network=args.tasks_per_network)
    run_all(args.data_dir, args.output_dir, config)
