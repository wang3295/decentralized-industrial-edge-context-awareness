from __future__ import annotations

import argparse
import copy
import json
import math
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, Set

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_joint_discovery_experiment import (
    CacheReplay,
    build_cache_replays,
    build_node_replays,
    environment_for_task,
)
from run_threshold_aware_mpc_experiment import (
    add_threshold_replay,
    calibrate_parameters,
)
from run_trust_robustness_experiment import (
    TrustExperimentConfig,
    bootstrap_ci,
    generate_trust_tasks,
    holm_adjust,
    initial_beta_state,
    joint_config,
    run_trust_ranked_search,
    search_config,
)

import sys

HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent
sys.path.insert(0, str(RAFT_DIR / "update_experiments"))
sys.path.insert(0, str(RAFT_DIR / "search_experiments"))

from run_update_experiments import load_inputs  # noqa: E402
from run_dynamic_search_experiments import create_connected_business_graph  # noqa: E402


MALICIOUS_RATES = (0.0, 0.2, 0.4)

PIPELINES = {
    "full": "Full: threshold MPC-DP + online trust + workflow",
    "original_mpc": "No threshold awareness: MPC-DP + online trust",
    "periodic": "Periodic update + online trust",
    "send_on_delta": "Send-on-delta + online trust",
    "no_trust": "No online trust",
    "gossip": "Gossip dissemination",
    "fresh_state": "Fresh-state upper bound",
}

PIPELINE_SPEC = {
    "full": ("threshold_mpc", "online_beta", "workflow"),
    "original_mpc": ("proposed", "online_beta", "workflow"),
    "periodic": ("fits_periodic", "online_beta", "workflow"),
    "send_on_delta": ("fits_delta", "online_beta", "workflow"),
    "no_trust": ("threshold_mpc", "no_trust", "workflow"),
    "gossip": ("threshold_mpc", "online_beta", "gossip"),
    "fresh_state": ("oracle_state", "online_beta", "workflow"),
}

COLORS = {
    "full": "#0072B2",
    "original_mpc": "#56B4E9",
    "periodic": "#E69F00",
    "send_on_delta": "#CC79A7",
    "no_trust": "#D55E00",
    "gossip": "#009E73",
    "fresh_state": "#666666",
}

MARKERS = {
    "full": "o",
    "original_mpc": "s",
    "periodic": "^",
    "send_on_delta": "v",
    "no_trust": "P",
    "gossip": "D",
    "fresh_state": "X",
}


@dataclass(frozen=True)
class FullAblationConfig(TrustExperimentConfig):
    network_repeats: int = 25
    tasks_per_network: int = 200


def cache_classification(
    actual_availability: np.ndarray,
    cached_availability: np.ndarray,
    demand: np.ndarray,
    origin: int,
    malicious_mask: np.ndarray,
) -> dict[str, float]:
    """Measure whether an update strategy preserves honest feasible candidates."""
    true_eligible = np.all(actual_availability >= demand, axis=1) & ~malicious_mask
    cached_eligible = np.all(cached_availability >= demand, axis=1) & ~malicious_mask
    true_eligible[origin] = False
    cached_eligible[origin] = False
    tp = int(np.sum(true_eligible & cached_eligible))
    fn = int(np.sum(true_eligible & ~cached_eligible))
    fp = int(np.sum(~true_eligible & cached_eligible))
    true_count = int(np.sum(true_eligible))
    cached_count = int(np.sum(cached_eligible))
    return {
        "true_honest_candidates": true_count,
        "cached_honest_candidates": cached_count,
        "true_positive_candidates": tp,
        "false_negative_candidates": fn,
        "false_positive_candidates": fp,
        "candidate_recall": tp / true_count if true_count else 1.0,
        "candidate_precision": tp / cached_count if cached_count else math.nan,
        "cache_blocked_discovery": int(tp == 0),
    }


def update_messages_per_request(
    replay: CacheReplay,
    graph: Sequence[Set[int]],
    tasks_per_network: int,
) -> float:
    total = sum(
        int(replay.actions[node].sum()) * len(graph[node])
        for node in range(len(graph))
    )
    return float(total / tasks_per_network)


def run_one_network_condition(
    malicious_rate: float,
    repeat: int,
    nodes_actual: np.ndarray,
    replays: dict[str, CacheReplay],
    cfg: FullAblationConfig,
    pipeline_labels: dict[str, str] | None = None,
    pipeline_spec: dict[str, tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    jcfg = joint_config(cfg)
    scfg = search_config(cfg)
    graph = create_connected_business_graph(
        cfg.num_nodes,
        cfg.average_degree,
        np.random.default_rng(50_000 + repeat),
    )

    malicious_count = int(round(malicious_rate * cfg.num_nodes))
    permutation = np.random.default_rng(60_000 + repeat).permutation(cfg.num_nodes)
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

    active_pipelines = PIPELINES if pipeline_labels is None else pipeline_labels
    active_spec = PIPELINE_SPEC if pipeline_spec is None else pipeline_spec
    if set(active_pipelines) != set(active_spec):
        raise ValueError("pipeline labels and specifications must have identical keys")

    rows: list[dict[str, float | int | str]] = []
    for pipeline in active_pipelines:
        replay_key, trust_policy, dissemination = active_spec[pipeline]
        replay = replays[replay_key]
        alpha = copy.deepcopy(initial_alpha)
        beta = copy.deepcopy(initial_beta)
        amortized_update_messages = update_messages_per_request(
            replay, graph, cfg.tasks_per_network
        )

        for task_idx, (time_slot, origin, demand) in enumerate(tasks):
            actual_availability = 100.0 - nodes_actual[:, time_slot]
            base_cached_availability = 100.0 - replay.cache_utilization[:, time_slot]
            cache_metrics = cache_classification(
                actual_availability,
                base_cached_availability,
                demand,
                origin,
                malicious_mask,
            )

            cached_availability = base_cached_availability.copy()
            if malicious_count:
                cached_availability[malicious_mask] = np.maximum(
                    cached_availability[malicious_mask],
                    demand + np.array([3.0, 0.5]),
                )
                cached_availability = np.clip(cached_availability, 0.0, 100.0)

            condition_seed = (
                int(round(10_000 * malicious_rate))
                + repeat * cfg.tasks_per_network
                + task_idx
            )
            behavior_valid = (
                np.random.default_rng(1_000_000 + condition_seed).random(cfg.num_nodes)
                < reliability
            )
            env = environment_for_task(
                graph,
                alpha,
                beta,
                actual_availability,
                cached_availability,
                origin,
                demand,
                jcfg,
                seed=2_000_000 + condition_seed,
            )
            result = run_trust_ranked_search(
                env,
                scfg,
                trust_policy,
                reliability,
                malicious_mask,
                behavior_valid,
                seed=3_000_000 + condition_seed,
                collection_ms=cfg.candidate_collection_ms,
                dissemination=dissemination,
            )
            if trust_policy in {"online_beta", "direct_only"}:
                alpha = copy.deepcopy(env.trust_alpha)
                beta = copy.deepcopy(env.trust_beta)

            rows.append(
                {
                    "malicious_rate": malicious_rate,
                    "repeat": repeat,
                    "task": task_idx,
                    "pipeline": pipeline,
                    "time_slot": time_slot,
                    "origin": origin,
                    "success": result.success,
                    "honest_success": result.honest_success,
                    "malicious_selected": result.malicious_selected,
                    "selected_candidate": result.selected_candidate,
                    "success_latency_ms": (
                        result.decision_latency_ms if result.success else math.nan
                    ),
                    "censored_latency_ms": result.decision_latency_ms,
                    "search_messages": result.message_count,
                    "request_messages": result.request_messages,
                    "response_messages": result.response_messages,
                    "confirmation_messages": result.confirmation_messages,
                    "duplicate_requests": result.duplicate_requests,
                    "selected_hops": result.selected_hops,
                    "amortized_update_messages": amortized_update_messages,
                    "total_messages": result.message_count
                    + amortized_update_messages,
                    "candidates_collected": result.candidates_collected,
                    "confirmation_attempts": result.confirmation_attempts,
                    "invalid_confirmations": result.invalid_confirmations,
                    "malicious_confirmations": result.malicious_confirmations,
                    "malicious_rejections": result.malicious_rejections,
                    "cache_wrmse": replay.cache_wrmse,
                    "update_rate": replay.update_rate,
                    **cache_metrics,
                }
            )
    return pd.DataFrame(rows)


def run_formal(
    nodes_actual: np.ndarray,
    replays: dict[str, CacheReplay],
    cfg: FullAblationConfig,
    workers: int,
    pipeline_labels: dict[str, str] | None = None,
    pipeline_spec: dict[str, tuple[str, str, str]] | None = None,
    repeat_offset: int = 0,
) -> pd.DataFrame:
    jobs = [
        (rate, repeat)
        for rate in MALICIOUS_RATES
        for repeat in range(repeat_offset, repeat_offset + cfg.network_repeats)
    ]
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_one_network_condition,
                rate,
                repeat,
                nodes_actual,
                replays,
                cfg,
                pipeline_labels,
                pipeline_spec,
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
    out = (
        raw.groupby(["malicious_rate", "repeat", "pipeline"], as_index=False)
        .agg(
            honest_success_rate=("honest_success", "mean"),
            any_success_rate=("success", "mean"),
            malicious_selection_rate=("malicious_selected", "mean"),
            cache_blocked_rate=("cache_blocked_discovery", "mean"),
            candidate_recall=("candidate_recall", "mean"),
            mean_search_messages=("search_messages", "mean"),
            mean_update_messages=("amortized_update_messages", "mean"),
            mean_total_messages=("total_messages", "mean"),
            mean_confirmations=("confirmation_attempts", "mean"),
            cache_wrmse=("cache_wrmse", "mean"),
            update_rate=("update_rate", "mean"),
        )
    )
    latency = (
        raw.groupby(["malicious_rate", "repeat", "pipeline"])["success_latency_ms"]
        .quantile(0.95)
        .rename("p95_success_latency_ms")
        .reset_index()
    )
    return out.merge(
        latency, on=["malicious_rate", "repeat", "pipeline"], how="left"
    )


def mean_and_ci(values: pd.Series) -> tuple[float, float]:
    x = values.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else math.nan, math.nan)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def summarize(network: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rate, pipeline), group in network.groupby(
        ["malicious_rate", "pipeline"], sort=False
    ):
        honest, honest_ci = mean_and_ci(100.0 * group.honest_success_rate)
        any_success, any_ci = mean_and_ci(100.0 * group.any_success_rate)
        malicious, malicious_ci = mean_and_ci(100.0 * group.malicious_selection_rate)
        blocked, blocked_ci = mean_and_ci(100.0 * group.cache_blocked_rate)
        rows.append(
            {
                "malicious_rate": rate,
                "pipeline": pipeline,
                "networks": len(group),
                "honest_success_percent": honest,
                "honest_success_ci95_percent": honest_ci,
                "any_success_percent": any_success,
                "any_success_ci95_percent": any_ci,
                "malicious_selection_percent": malicious,
                "malicious_selection_ci95_percent": malicious_ci,
                "cache_blocked_percent": blocked,
                "cache_blocked_ci95_percent": blocked_ci,
                "candidate_recall_percent": 100.0 * float(group.candidate_recall.mean()),
                "p95_success_latency_ms": float(group.p95_success_latency_ms.mean()),
                "mean_search_messages": float(group.mean_search_messages.mean()),
                "mean_update_messages": float(group.mean_update_messages.mean()),
                "mean_total_messages": float(group.mean_total_messages.mean()),
                "mean_confirmations": float(group.mean_confirmations.mean()),
                "cache_wrmse": float(group.cache_wrmse.mean()),
                "update_rate_percent": 100.0 * float(group.update_rate.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["malicious_rate", "pipeline"])


def paired_tests(network: pd.DataFrame) -> pd.DataFrame:
    comparators = ("original_mpc", "periodic", "send_on_delta", "no_trust", "gossip")
    endpoints = ("honest_success_rate", "malicious_selection_rate")
    rows = []
    for rate_idx, rate in enumerate(MALICIOUS_RATES):
        full = network[
            np.isclose(network.malicious_rate, rate) & network.pipeline.eq("full")
        ][["repeat", *endpoints]]
        for comparator_idx, comparator in enumerate(comparators):
            baseline = network[
                np.isclose(network.malicious_rate, rate)
                & network.pipeline.eq(comparator)
            ][["repeat", *endpoints]]
            merged = full.merge(baseline, on="repeat", suffixes=("_full", "_baseline"))
            for endpoint_idx, endpoint in enumerate(endpoints):
                diff = 100.0 * (
                    merged[f"{endpoint}_full"].to_numpy()
                    - merged[f"{endpoint}_baseline"].to_numpy()
                )
                p_value = (
                    1.0
                    if np.allclose(diff, 0.0)
                    else float(
                        wilcoxon(
                            diff, zero_method="pratt", alternative="two-sided"
                        ).pvalue
                    )
                )
                low, high = bootstrap_ci(
                    diff,
                    seed=140_000
                    + rate_idx * 1_000
                    + comparator_idx * 100
                    + endpoint_idx,
                )
                rows.append(
                    {
                        "malicious_rate": rate,
                        "endpoint": endpoint,
                        "comparison": f"full vs {comparator}",
                        "network_clusters": len(diff),
                        "full_percent": 100.0
                        * float(merged[f"{endpoint}_full"].mean()),
                        "baseline_percent": 100.0
                        * float(merged[f"{endpoint}_baseline"].mean()),
                        "full_minus_baseline_pp": float(diff.mean()),
                        "cluster_bootstrap_ci_low_pp": low,
                        "cluster_bootstrap_ci_high_pp": high,
                        "wilcoxon_p": p_value,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p"] = holm_adjust(frame.wilcoxon_p)
    return frame


def cache_blocking_tests(network: pd.DataFrame) -> pd.DataFrame:
    """Secondary, separately corrected update-policy comparison family."""
    rows = []
    comparators = ("original_mpc", "periodic", "send_on_delta")
    for rate_idx, rate in enumerate(MALICIOUS_RATES):
        full = network[
            np.isclose(network.malicious_rate, rate) & network.pipeline.eq("full")
        ][["repeat", "cache_blocked_rate"]]
        for comparator_idx, comparator in enumerate(comparators):
            baseline = network[
                np.isclose(network.malicious_rate, rate)
                & network.pipeline.eq(comparator)
            ][["repeat", "cache_blocked_rate"]]
            merged = full.merge(baseline, on="repeat", suffixes=("_full", "_baseline"))
            diff = 100.0 * (
                merged.cache_blocked_rate_full.to_numpy()
                - merged.cache_blocked_rate_baseline.to_numpy()
            )
            p_value = (
                1.0
                if np.allclose(diff, 0.0)
                else float(
                    wilcoxon(diff, zero_method="pratt", alternative="two-sided").pvalue
                )
            )
            low, high = bootstrap_ci(
                diff, seed=180_000 + rate_idx * 100 + comparator_idx
            )
            rows.append(
                {
                    "malicious_rate": rate,
                    "comparison": f"full vs {comparator}",
                    "network_clusters": len(diff),
                    "full_cache_blocked_percent": 100.0
                    * float(merged.cache_blocked_rate_full.mean()),
                    "baseline_cache_blocked_percent": 100.0
                    * float(merged.cache_blocked_rate_baseline.mean()),
                    "full_minus_baseline_pp": float(diff.mean()),
                    "cluster_bootstrap_ci_low_pp": low,
                    "cluster_bootstrap_ci_high_pp": high,
                    "wilcoxon_p": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p"] = holm_adjust(frame.wilcoxon_p)
    return frame


def export_figure(fig: plt.Figure, output_dir: Path) -> None:
    stem = output_dir / "full_pipeline_ablation"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    rate_percent = 100.0 * np.asarray(MALICIOUS_RATES)

    for pipeline in PIPELINES:
        data = summary[summary.pipeline.eq(pipeline)].sort_values("malicious_rate")
        linestyle = "--" if pipeline == "fresh_state" else "-"
        axes[0, 0].errorbar(
            rate_percent,
            data.honest_success_percent,
            yerr=data.honest_success_ci95_percent,
            color=COLORS[pipeline],
            marker=MARKERS[pipeline],
            linestyle=linestyle,
            linewidth=1.35,
            markersize=4.0,
            capsize=2,
            label=PIPELINES[pipeline],
        )

    for pipeline in ("full", "no_trust", "gossip", "fresh_state"):
        data = summary[summary.pipeline.eq(pipeline)].sort_values("malicious_rate")
        axes[0, 1].errorbar(
            rate_percent,
            data.malicious_selection_percent,
            yerr=data.malicious_selection_ci95_percent,
            color=COLORS[pipeline],
            marker=MARKERS[pipeline],
            linestyle="--" if pipeline == "fresh_state" else "-",
            linewidth=1.4,
            markersize=4.0,
            capsize=2,
            label=PIPELINES[pipeline],
        )

    at_40 = summary[np.isclose(summary.malicious_rate, 0.4)].set_index("pipeline")
    update_methods = ["full", "original_mpc", "periodic", "send_on_delta", "fresh_state"]
    x = np.arange(len(update_methods))
    axes[1, 0].bar(
        x,
        at_40.loc[update_methods, "cache_blocked_percent"],
        yerr=at_40.loc[update_methods, "cache_blocked_ci95_percent"],
        color=[COLORS[p] for p in update_methods],
        edgecolor="white",
        linewidth=0.5,
        capsize=2,
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(
        ["Full", "Original\nMPC-DP", "Periodic", "Delta", "Fresh\nstate"]
    )

    for pipeline in PIPELINES:
        row = at_40.loc[pipeline]
        axes[1, 1].scatter(
            row.mean_total_messages,
            row.honest_success_percent,
            s=38,
            marker=MARKERS[pipeline],
            color=COLORS[pipeline],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        label = {
            "full": "Full",
            "original_mpc": "Original MPC",
            "periodic": "Periodic",
            "send_on_delta": "Delta",
            "no_trust": "No trust",
            "gossip": "Gossip",
            "fresh_state": "Fresh state",
        }[pipeline]
        offsets = {
            "full": (6, 4),
            "original_mpc": (-54, -8),
            "periodic": (6, 9),
            "send_on_delta": (6, -4),
            "no_trust": (6, 3),
            "gossip": (6, 3),
            "fresh_state": (4, 4),
        }
        axes[1, 1].annotate(
            label,
            (row.mean_total_messages, row.honest_success_percent),
            xytext=offsets[pipeline],
            textcoords="offset points",
            fontsize=5.8,
        )

    axes[0, 0].set_title("(a) Trustworthy candidate discovery", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Malicious-node fraction (%)")
    axes[0, 0].set_ylabel("Honest candidate discovery (%)")
    axes[0, 1].set_title("(b) Malicious candidate selection", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Malicious-node fraction (%)")
    axes[0, 1].set_ylabel("Malicious selection (%)")
    axes[1, 0].set_title("(c) Cache-induced blocking at 40% malicious", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Requests with no cached honest candidate (%)")
    axes[1, 1].set_title("(d) Discovery–communication trade-off at 40%", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Search + amortized update messages/request")
    axes[1, 1].set_ylabel("Honest candidate discovery (%)")
    for ax in axes.flat:
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    export_figure(fig, output_dir)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    data = frame[list(columns)]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in data.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("NA")
            elif isinstance(value, (float, np.floating)):
                number = float(value)
                values.append(
                    f"{number:.3e}" if 0 < abs(number) < 1e-4 else f"{number:.4f}"
                )
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def value(summary: pd.DataFrame, rate: float, pipeline: str, field: str) -> float:
    row = summary[np.isclose(summary.malicious_rate, rate) & summary.pipeline.eq(pipeline)]
    return float(row.iloc[0][field])


def write_report(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    cache_tests: pd.DataFrame,
    cfg: FullAblationConfig,
    output_dir: Path,
) -> None:
    def test_value(
        rate_value: float, endpoint: str, comparison: str, field: str
    ) -> float:
        row = tests[
            np.isclose(tests.malicious_rate, rate_value)
            & tests.endpoint.eq(endpoint)
            & tests.comparison.eq(comparison)
        ]
        return float(row.iloc[0][field])

    rate = 0.4
    full_honest = value(summary, rate, "full", "honest_success_percent")
    no_trust_honest = value(summary, rate, "no_trust", "honest_success_percent")
    gossip_honest = value(summary, rate, "gossip", "honest_success_percent")
    fresh_honest = value(summary, rate, "fresh_state", "honest_success_percent")
    full_mal = value(summary, rate, "full", "malicious_selection_percent")
    no_trust_mal = value(summary, rate, "no_trust", "malicious_selection_percent")
    full_blocked = value(summary, rate, "full", "cache_blocked_percent")
    original_blocked = value(summary, rate, "original_mpc", "cache_blocked_percent")
    periodic_blocked = value(summary, rate, "periodic", "cache_blocked_percent")
    delta_blocked = value(summary, rate, "send_on_delta", "cache_blocked_percent")
    full_messages = value(summary, rate, "full", "mean_total_messages")
    gossip_messages = value(summary, rate, "gossip", "mean_total_messages")
    gossip_reduction = 100.0 * (full_messages - gossip_messages) / full_messages
    full_wrmse = value(summary, rate, "full", "cache_wrmse")
    periodic_wrmse = value(summary, rate, "periodic", "cache_wrmse")
    original_gain = test_value(
        rate,
        "honest_success_rate",
        "full vs original_mpc",
        "full_minus_baseline_pp",
    )
    original_p = test_value(
        rate, "honest_success_rate", "full vs original_mpc", "holm_adjusted_p"
    )
    periodic_gap = test_value(
        rate,
        "honest_success_rate",
        "full vs periodic",
        "full_minus_baseline_pp",
    )
    periodic_p = test_value(
        rate, "honest_success_rate", "full vs periodic", "holm_adjusted_p"
    )
    delta_gain = test_value(
        rate,
        "honest_success_rate",
        "full vs send_on_delta",
        "full_minus_baseline_pp",
    )
    delta_p = test_value(
        rate, "honest_success_rate", "full vs send_on_delta", "holm_adjusted_p"
    )
    gossip_gain = test_value(
        rate,
        "honest_success_rate",
        "full vs gossip",
        "full_minus_baseline_pp",
    )
    gossip_p = test_value(
        rate, "honest_success_rate", "full vs gossip", "holm_adjusted_p"
    )

    report = f"""# 卸载前可信候选节点发现：全链路组件消融实验

## 1. 实验回答的问题

本实验不执行具体任务卸载，而是评价从资源状态维护到可信候选节点输出的完整卸载前流程。完整方法为“门槛感知FITS-MPC-DP更新 + 业务关系多跳搜索 + 在线Beta直接信任/路径信任 + top-k二次确认”。消融实验逐一替换更新、信任和传播组件，判断三部分是否只是相互独立的实验，还是会共同影响最终候选发现。

## 2. 冻结实验协议

- 每个条件包含{cfg.network_repeats}个独立网络，每个网络连续处理{cfg.tasks_per_network}个远端必需任务；网络拓扑是独立统计单位，单个任务仅是网络内重复观测。
- 节点数{cfg.num_nodes}，平均业务度数{cfg.average_degree}，最大转发深度{cfg.max_hops}，候选收集截止{cfg.candidate_collection_ms:g} ms，绝对截止{cfg.deadline_ms:g} ms，top-k={cfg.top_k}。
- 更新预算固定为{cfg.update_budget:.0%}；完整方法eta={cfg.eta:g}、门槛惩罚lambda={cfg.threshold_lambda:g}、CPU/内存权重0.5/0.5。
- 恶意节点比例为0%、20%和40%，诚实/恶意行为可靠率为{cfg.honest_reliability:.0%}/{cfg.malicious_reliability:.0%}；恶意节点可虚假声明资源，确认阶段观察实际资源和行为结果。
- 同一网络内所有方法共享资源轨迹、任务、拓扑、节点故障、链路抖动、恶意集合与行为随机数。
- Fresh-state每时隙发送真实状态，更新率100%，不满足相同50%预算，仅作为不可部署的状态新鲜度上界。

## 3. 完整结果

{markdown_table(summary, ['malicious_rate', 'pipeline', 'networks', 'honest_success_percent', 'honest_success_ci95_percent', 'malicious_selection_percent', 'malicious_selection_ci95_percent', 'cache_blocked_percent', 'candidate_recall_percent', 'p95_success_latency_ms', 'mean_search_messages', 'mean_update_messages', 'mean_total_messages', 'cache_wrmse', 'update_rate_percent'])}

## 4. 40%恶意节点下的重点读数

- 完整方法的诚实候选发现率为{full_honest:.2f}%，无在线信任时为{no_trust_honest:.2f}%，Gossip传播时为{gossip_honest:.2f}%，Fresh-state上界为{fresh_honest:.2f}%。
- 完整方法的恶意候选误选率为{full_mal:.2f}%，无在线信任时为{no_trust_mal:.2f}%。
- 完整方法因缓存漏掉全部诚实候选而阻塞发现的比例为{full_blocked:.2f}%；原MPC-DP、周期更新和Send-on-delta分别为{original_blocked:.2f}%、{periodic_blocked:.2f}%和{delta_blocked:.2f}%。
- 各差异必须与下表的网络级效应量、95%聚类bootstrap置信区间和Holm校正p值共同解释。

## 5. 网络级配对推断

{markdown_table(tests, ['malicious_rate', 'endpoint', 'comparison', 'network_clusters', 'full_percent', 'baseline_percent', 'full_minus_baseline_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_p', 'holm_adjusted_p'])}

### 5.1 更新策略的缓存阻塞率（独立的次要比较族）

{markdown_table(cache_tests, ['malicious_rate', 'comparison', 'network_clusters', 'full_cache_blocked_percent', 'baseline_cache_blocked_percent', 'full_minus_baseline_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_p', 'holm_adjusted_p'])}

## 6. 结果应如何解释

1. 更新机制首先决定真实可用节点是否仍出现在资源缓存的候选集合中。40%恶意节点时，完整方法相对原MPC-DP把诚实发现率提高{original_gain:.2f}个百分点（Holm校正p={original_p:.4g}），并把缓存阻塞率由{original_blocked:.2f}%降至{full_blocked:.2f}%。
2. 相对Send-on-delta，完整方法在40%条件下的发现率高{delta_gain:.2f}个百分点，但覆盖全部主终点比较后的p={delta_p:.4g}，未达到0.05；缓存阻塞率的独立次要检验则支持完整方法优于Delta。
3. 周期更新的最终发现率比完整方法高{-periodic_gap:.2f}个百分点（Holm校正p={periodic_p:.4g}），且缓存阻塞更低，但其连续缓存WRMSE为{periodic_wrmse:.4f}，明显高于完整方法的{full_wrmse:.4f}。因此结果支持两种更新目标之间的折衷，不支持“完整方法全面优于周期更新”。
4. 在线信任不改变缓存，却改变有限top-k确认名额的排序。40%恶意节点时，它把诚实发现率从{no_trust_honest:.2f}%提高到{full_honest:.2f}%，同时把恶意误选率从{no_trust_mal:.2f}%降至{full_mal:.2f}%。
5. 业务关系全转发相对Gossip在40%条件下提高{gossip_gain:.2f}个百分点的诚实发现率（Holm校正p={gossip_p:.4g}），代价是总消息量由{gossip_messages:.2f}增至{full_messages:.2f}条/请求；Gossip节省约{gossip_reduction:.2f}%通信量。这是覆盖率与通信开销的明确折衷。
6. Fresh-state不是公平基线，其作用是显示有限更新预算相对完美状态视图仍有多大性能差距。

## 7. 结论边界

- 本实验的终点是“返回经确认的可信可用候选节点”，不是任务真正执行、排队、迁移或端到端完成时间。
- 结果来自真实资源轨迹驱动的动态网络仿真，不替代嵌入式ARM或真实工业网络部署。
- 攻击模型覆盖虚假资源广告与低可靠响应，不覆盖串谋、身份伪造、on-off攻击或任务执行后的隐私攻击。
- 统计推断以独立网络为n={cfg.network_repeats}，没有把{cfg.tasks_per_network}次网络内请求伪装成独立样本。
"""
    (output_dir / "全链路组件消融实验结果与分析.md").write_text(
        report, encoding="utf-8"
    )


def write_figure_contract(
    output_dir: Path, cfg: FullAblationConfig, raw: pd.DataFrame
) -> None:
    total_rows = len(raw)
    successful_rows = int(raw.success.sum())
    failed_rows = total_rows - successful_rows
    undefined_precision = int(raw.candidate_precision.isna().sum())
    text = f"""# Figure contract and QA

- Core conclusion: threshold-aware updating, online trust learning, and workflow-constrained dissemination affect distinct bottlenecks in trustworthy pre-offloading candidate discovery.
- Figure archetype: quantitative grid with panel (a) as the hero evidence.
- Target/output: double-column manuscript figure, 182.9 mm wide, editable SVG/PDF plus 300-dpi PNG and 600-dpi TIFF.
- Backend: Python (matplotlib) exclusively.
- Panel a: honest candidate discovery across malicious-node fractions.
- Panel b: malicious candidate selection for trust/search controls.
- Panel c: cache-induced blocking for update-policy ablations at 40% malicious nodes.
- Panel d: discovery–communication trade-off at 40% malicious nodes.
- Independent unit: one generated network topology (n={cfg.network_repeats} per condition); {cfg.tasks_per_network} sequential requests are repeated observations within a network.
- Error bars: 95% normal-approximation confidence intervals over network-level means.
- Inference: paired two-sided Wilcoxon signed-rank tests on network-level endpoint differences; all planned full-versus-ablation comparisons for both primary endpoints and all malicious fractions are one Holm family.
- Data integrity: all generated rows are retained; no networks, requests, pipelines, or endpoints are excluded.
- Missing-data audit: all {total_rows} request rows enter the primary success and malicious-selection analyses. Success-conditioned latency is defined only for the {successful_rows} successful requests; the {failed_rows} failed requests retain the prespecified {cfg.deadline_ms:g}-ms censored latency but have no success latency. Candidate precision is undefined for {undefined_precision} rows with an empty cached honest-candidate set; these rows remain in success, blocking and recall analyses.
- Simulation boundary: resource traces are observed/replayed data, whereas network topology, tasks, failures, malicious behavior and communication delays are generated by the frozen simulation protocol. No panel is presented as hardware-deployment evidence.
- Reviewer risk: Fresh-state has a 100% update rate and must be labeled an upper bound, not a budget-matched baseline. Simulation does not establish real-hardware latency.
"""
    (output_dir / "figure_contract_and_qa.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "full_pipeline_ablation_formal",
    )
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--tasks-per-network", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = FullAblationConfig(
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
            f"Frozen eta/lambda={cfg.eta}/{cfg.threshold_lambda}, "
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
    required_replays = {spec[0] for spec in PIPELINE_SPEC.values()}
    missing = required_replays - set(replays)
    if missing:
        raise AssertionError(f"Missing cache replays: {sorted(missing)}")

    raw = run_formal(nodes_actual, replays, cfg, args.workers)
    network = network_summary(raw)
    summary = summarize(network)
    tests = paired_tests(network)
    cache_tests = cache_blocking_tests(network)

    raw.to_csv(args.output_dir / "full_pipeline_ablation_raw.csv", index=False)
    network.to_csv(args.output_dir / "full_pipeline_ablation_network.csv", index=False)
    summary.to_csv(args.output_dir / "full_pipeline_ablation_summary.csv", index=False)
    tests.to_csv(args.output_dir / "full_pipeline_ablation_network_tests.csv", index=False)
    cache_tests.to_csv(
        args.output_dir / "full_pipeline_ablation_cache_tests.csv", index=False
    )
    calibration.to_csv(args.output_dir / "frozen_threshold_calibration.csv", index=False)
    plot_results(summary, args.output_dir)
    write_report(summary, tests, cache_tests, cfg, args.output_dir)
    write_figure_contract(args.output_dir, cfg, raw)

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(cfg),
        "malicious_rates": list(MALICIOUS_RATES),
        "pipelines": PIPELINES,
        "pipeline_spec": PIPELINE_SPEC,
        "independent_unit": "network topology repeat",
        "within_network_repeated_observations": cfg.tasks_per_network,
        "primary_endpoints": [
            "honest candidate discovery rate",
            "malicious candidate selection rate",
        ],
        "multiple_comparison_family": "30 paired network-level comparisons; Holm correction",
        "secondary_multiple_comparison_family": "9 cache-blocking comparisons; separate Holm correction",
        "fresh_state_boundary": "100% update upper bound; not budget matched",
        "selected_eta": selected_eta,
        "selected_threshold_lambda": selected_lambda,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nFull-pipeline ablation summary\n")
    print(
        summary[
            [
                "malicious_rate",
                "pipeline",
                "honest_success_percent",
                "malicious_selection_percent",
                "cache_blocked_percent",
                "mean_total_messages",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
