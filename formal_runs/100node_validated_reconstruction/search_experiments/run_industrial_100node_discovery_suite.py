from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent
sys.path.insert(0, str(RAFT_DIR / "joint_experiments"))
sys.path.insert(0, str(RAFT_DIR / "update_experiments"))

from run_full_pipeline_ablation import (  # noqa: E402
    FullAblationConfig,
    run_one_network_condition,
)
from run_joint_discovery_experiment import (  # noqa: E402
    build_node_replays,
)
from run_deployment_equivalent_network_scale import (  # noqa: E402
    build_deployment_replays,
)
from run_trust_robustness_experiment import (  # noqa: E402
    bootstrap_ci,
    holm_adjust,
    joint_config,
)
from run_update_experiments import load_inputs  # noqa: E402


PROTOCOL_LABELS = {
    "proposed": "Proposed Workflow",
    "gossip": "Gossip, fanout=2",
    "random_walk": "Random Walk",
    "expanding_ring": "Expanding Ring Search",
    "centralized_oracle": "Centralized Oracle",
}

PROTOCOL_SPEC = {
    "proposed": ("runtime_mpc_dp", "online_beta", "workflow"),
    "gossip": ("runtime_mpc_dp", "online_beta", "gossip"),
    "random_walk": ("runtime_mpc_dp", "online_beta", "random_walk"),
    "expanding_ring": ("runtime_mpc_dp", "online_beta", "expanding_ring"),
    "centralized_oracle": ("runtime_mpc_dp", "oracle", "centralized_oracle"),
}

DECENTRALIZED_LABELS = {
    key: value for key, value in PROTOCOL_LABELS.items() if key != "centralized_oracle"
}
DECENTRALIZED_SPEC = {key: PROTOCOL_SPEC[key] for key in DECENTRALIZED_LABELS}

TRUST_LABELS = {
    "path_trust": "Online Beta + path trust",
    "direct_only": "Online Beta, direct only",
    "no_trust": "No trust",
    "no_confirmation": "Path trust, no confirmation",
    "oracle_trust": "Oracle trust",
}

TRUST_SPEC = {
    "path_trust": ("runtime_mpc_dp", "online_beta", "workflow"),
    "direct_only": ("runtime_mpc_dp", "direct_only", "workflow"),
    "no_trust": ("runtime_mpc_dp", "no_trust", "workflow"),
    "no_confirmation": ("runtime_mpc_dp", "online_beta", "workflow_no_confirmation"),
    "oracle_trust": ("runtime_mpc_dp", "oracle", "workflow"),
}

COLORS = {
    "proposed": "#0072B2",
    "gossip": "#009E73",
    "random_walk": "#CC79A7",
    "expanding_ring": "#E69F00",
    "centralized_oracle": "#666666",
    "path_trust": "#0072B2",
    "direct_only": "#56B4E9",
    "no_trust": "#777777",
    "no_confirmation": "#D55E00",
    "oracle_trust": "#E69F00",
}

MARKERS = {
    "proposed": "o",
    "gossip": "s",
    "random_walk": "^",
    "expanding_ring": "D",
    "centralized_oracle": "X",
    "path_trust": "o",
    "direct_only": "v",
    "no_trust": "^",
    "no_confirmation": "s",
    "oracle_trust": "D",
}

ROBUSTNESS_LEVELS = {
    "outage_rate": (0.0, 0.05, 0.10, 0.20),
    "topology_churn_rate": (0.0, 0.05, 0.10, 0.20),
    "resource_change_probability": (0.0, 0.10, 0.30, 0.50),
}

HYPER_LEVELS = {
    "max_hops": (2.0, 3.0, 4.0, 5.0, 6.0),
    "deadline_ms": (90.0, 120.0, 150.0, 180.0, 240.0),
    "top_k": (1.0, 2.0, 3.0, 4.0, 5.0),
    "gossip_fanout": (1.0, 2.0, 3.0, 4.0),
}

REFERENCE_LEVEL = {
    "max_hops": 5.0,
    "deadline_ms": 180.0,
    "top_k": 3.0,
    "gossip_fanout": 2.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_config(repeats: int, tasks: int, trace_length: int) -> FullAblationConfig:
    return FullAblationConfig(
        num_nodes=100,
        trace_length=trace_length,
        trace_start=94,
        offset_step=1,
        average_degree=4,
        update_budget=0.5,
        horizon=6,
        eta=1.0,
        # The threshold-aware extension is not part of the paper's original
        # MPC-DP.  Keep the dataclass field neutral and use the shared deployed
        # implementation below so this suite matches the physical testbed and
        # the 50/100/150-node scale experiment.
        threshold_lambda=0.0,
        tasks_per_network=tasks,
        network_repeats=repeats,
        outage_rate=0.10,
        topology_churn_rate=0.10,
        resource_change_probability=0.30,
        resource_shock_std=11.0,
        max_hops=5,
        deadline_ms=180.0,
        candidate_collection_ms=120.0,
        top_k=3,
        gossip_fanout=2,
        random_walk_restarts=3,
        expanding_ring_interval_ms=24.0,
    )


def build_inputs(data_dir: Path, cfg: FullAblationConfig):
    actual, predictions = load_inputs(data_dir)
    jcfg = joint_config(cfg)
    nodes_actual, nodes_predictions = build_node_replays(actual, predictions, jcfg)
    replays = build_deployment_replays(
        nodes_actual,
        nodes_predictions,
        cfg,
        cfg.eta,
        delta_threshold=5.0,
    )
    mpc_replay = replays["runtime_mpc_dp"]
    if not np.isclose(mpc_replay.update_rate, cfg.update_budget):
        raise AssertionError(
            f"MPC-DP update rate {mpc_replay.update_rate} does not match "
            f"budget {cfg.update_budget}"
        )
    return nodes_actual, replays


def run_block(
    output_dir: Path,
    block_name: str,
    experiment: str,
    factor: str,
    level: float,
    malicious_rate: float,
    cfg: FullAblationConfig,
    labels: dict[str, str],
    specifications: dict[str, tuple[str, str, str]],
    nodes_actual: np.ndarray,
    replays: dict,
    workers: int,
) -> pd.DataFrame:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{block_name}.csv"
    expected = cfg.network_repeats * cfg.tasks_per_network * len(labels)
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        if len(existing) == expected:
            print(f"[reuse] {block_name}: {len(existing)} rows", flush=True)
            return existing

    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                run_one_network_condition,
                malicious_rate,
                repeat,
                nodes_actual,
                replays,
                cfg,
                labels,
                specifications,
            ): repeat
            for repeat in range(cfg.network_repeats)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            repeat = futures[future]
            frame = future.result()
            frame.insert(0, "experiment", experiment)
            frame.insert(1, "factor", factor)
            frame.insert(2, "level", float(level))
            frames.append(frame)
            if completed % 5 == 0 or completed == len(futures):
                print(
                    f"[{block_name}] {completed:02d}/{len(futures):02d}; repeat={repeat}",
                    flush=True,
                )
    raw = pd.concat(frames, ignore_index=True)
    if len(raw) != expected:
        raise AssertionError(f"{block_name}: expected {expected} rows, got {len(raw)}")
    raw.to_csv(checkpoint, index=False)
    return raw


def network_summary(raw: pd.DataFrame) -> pd.DataFrame:
    def finite_median(values: pd.Series) -> float:
        values = values[np.isfinite(values)]
        return float(values.median()) if len(values) else math.nan

    def finite_p95(values: pd.Series) -> float:
        values = values[np.isfinite(values)]
        return float(values.quantile(0.95)) if len(values) else math.nan

    def finite_mean(values: pd.Series) -> float:
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else math.nan

    grouped = raw.groupby(
        ["experiment", "factor", "level", "malicious_rate", "pipeline", "repeat"],
        as_index=False,
        dropna=False,
    )
    return grouped.agg(
        success_rate=("success", "mean"),
        honest_success_rate=("honest_success", "mean"),
        malicious_selection_rate=("malicious_selected", "mean"),
        median_success_latency_ms=("success_latency_ms", finite_median),
        p95_success_latency_ms=("success_latency_ms", finite_p95),
        mean_censored_latency_ms=("censored_latency_ms", "mean"),
        mean_discovery_messages=("search_messages", "mean"),
        mean_request_messages=("request_messages", "mean"),
        mean_response_messages=("response_messages", "mean"),
        mean_confirmation_messages=("confirmation_messages", "mean"),
        mean_duplicate_requests=("duplicate_requests", "mean"),
        mean_confirmations=("confirmation_attempts", "mean"),
        mean_invalid_confirmations=("invalid_confirmations", "mean"),
        mean_selected_hops=("selected_hops", finite_mean),
        mean_candidates=("candidates_collected", "mean"),
    )


def cluster_mean_ci(values: Iterable[float], seed: int) -> tuple[float, float, float]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return math.nan, math.nan, math.nan
    if len(data) == 1:
        return float(data[0]), math.nan, math.nan
    low, high = bootstrap_ci(data, seed=seed, draws=5_000)
    return float(data.mean()), low, high


def summarize(network: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["experiment", "factor", "level", "malicious_rate", "pipeline"]
    metrics = (
        "success_rate",
        "honest_success_rate",
        "malicious_selection_rate",
        "median_success_latency_ms",
        "p95_success_latency_ms",
        "mean_censored_latency_ms",
        "mean_discovery_messages",
        "mean_request_messages",
        "mean_response_messages",
        "mean_confirmation_messages",
        "mean_duplicate_requests",
        "mean_confirmations",
        "mean_invalid_confirmations",
        "mean_selected_hops",
        "mean_candidates",
    )
    for index, (group_key, frame) in enumerate(network.groupby(keys, sort=False, dropna=False)):
        row = dict(zip(keys, group_key))
        row["networks"] = int(frame.repeat.nunique())
        for metric_idx, metric in enumerate(metrics):
            scale = 100.0 if metric.endswith("_rate") else 1.0
            mean, low, high = cluster_mean_ci(
                scale * frame[metric].to_numpy(),
                seed=710_000 + index * 100 + metric_idx,
            )
            output_name = metric.replace("_rate", "_percent") if scale == 100.0 else metric
            row[output_name] = mean
            row[f"{output_name}_ci_low"] = low
            row[f"{output_name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(
    network: pd.DataFrame,
    experiment: str,
    proposed: str,
    comparators: Sequence[str],
    endpoints: Sequence[str],
    seed_offset: int,
) -> pd.DataFrame:
    subset = network[network.experiment.eq(experiment)]
    rows = []
    group_cols = ["factor", "level", "malicious_rate"]
    for group_idx, (condition, frame) in enumerate(
        subset.groupby(group_cols, sort=False, dropna=False)
    ):
        reference = frame[frame.pipeline.eq(proposed)][["repeat", *endpoints]]
        if reference.empty:
            continue
        for comparator_idx, comparator in enumerate(comparators):
            baseline = frame[frame.pipeline.eq(comparator)][["repeat", *endpoints]]
            if baseline.empty:
                continue
            pair = reference.merge(
                baseline,
                on="repeat",
                suffixes=("_proposed", "_baseline"),
                validate="one_to_one",
            )
            for endpoint_idx, endpoint in enumerate(endpoints):
                scale = 100.0 if endpoint.endswith("_rate") else 1.0
                diff = scale * (
                    pair[f"{endpoint}_proposed"].to_numpy()
                    - pair[f"{endpoint}_baseline"].to_numpy()
                )
                diff = diff[np.isfinite(diff)]
                if not len(diff):
                    continue
                p_value = (
                    1.0
                    if np.allclose(diff, 0.0)
                    else float(
                        wilcoxon(
                            diff,
                            zero_method="pratt",
                            alternative="two-sided",
                            method="auto",
                        ).pvalue
                    )
                )
                low, high = bootstrap_ci(
                    diff,
                    seed=seed_offset
                    + group_idx * 1_000
                    + comparator_idx * 100
                    + endpoint_idx,
                    draws=5_000,
                )
                rows.append(
                    {
                        "experiment": experiment,
                        "factor": condition[0],
                        "level": condition[1],
                        "malicious_rate": condition[2],
                        "endpoint": endpoint,
                        "comparison": f"{proposed} vs {comparator}",
                        "network_clusters": len(diff),
                        "proposed_minus_baseline": float(diff.mean()),
                        "cluster_bootstrap_ci_low": low,
                        "cluster_bootstrap_ci_high": high,
                        "wilcoxon_p": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if len(result):
        result["holm_adjusted_p"] = holm_adjust(result.wilcoxon_p.to_numpy())
    return result


def hyperparameter_tests(network: pd.DataFrame) -> pd.DataFrame:
    subset = network[network.experiment.eq("hyperparameter")]
    rows = []
    for factor_idx, (factor, frame) in enumerate(subset.groupby("factor", sort=False)):
        reference_level = REFERENCE_LEVEL[factor]
        ref = frame[np.isclose(frame.level, reference_level)][
            ["repeat", "success_rate", "mean_discovery_messages"]
        ]
        for level_idx, level in enumerate(sorted(frame.level.unique())):
            if np.isclose(level, reference_level):
                continue
            alt = frame[np.isclose(frame.level, level)][
                ["repeat", "success_rate", "mean_discovery_messages"]
            ]
            pair = ref.merge(alt, on="repeat", suffixes=("_reference", "_alternative"))
            for endpoint_idx, endpoint in enumerate(
                ("success_rate", "mean_discovery_messages")
            ):
                scale = 100.0 if endpoint == "success_rate" else 1.0
                diff = scale * (
                    pair[f"{endpoint}_alternative"].to_numpy()
                    - pair[f"{endpoint}_reference"].to_numpy()
                )
                p_value = (
                    1.0
                    if np.allclose(diff, 0.0)
                    else float(wilcoxon(diff, zero_method="pratt").pvalue)
                )
                low, high = bootstrap_ci(
                    diff,
                    seed=850_000 + factor_idx * 100 + level_idx * 10 + endpoint_idx,
                )
                rows.append(
                    {
                        "factor": factor,
                        "reference_level": reference_level,
                        "alternative_level": level,
                        "endpoint": endpoint,
                        "network_clusters": len(diff),
                        "alternative_minus_reference": float(diff.mean()),
                        "cluster_bootstrap_ci_low": low,
                        "cluster_bootstrap_ci_high": high,
                        "wilcoxon_p": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = holm_adjust(result.wilcoxon_p.to_numpy())
    return result


def configure_plotting() -> None:
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
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def export_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        output_dir / f"{stem}.tiff",
        dpi=600,
    )


def errorbar(ax, data: pd.DataFrame, x: str, y: str, key: str, label: str) -> None:
    data = data.sort_values(x)
    lower = data[y] - data[f"{y}_ci_low"]
    upper = data[f"{y}_ci_high"] - data[y]
    ax.errorbar(
        data[x],
        data[y],
        yerr=np.vstack([lower, upper]),
        color=COLORS[key],
        marker=MARKERS[key],
        linewidth=1.35,
        markersize=4.0,
        capsize=2,
        label=label,
    )


def plot_protocol(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[summary.experiment.eq("protocol_main")].copy()
    order = list(PROTOCOL_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    x = np.arange(len(order))
    for ax, field, title, ylabel in (
        (axes[0], "honest_success_percent", "(a) Trustworthy discovery", "Success (%)"),
        (axes[1], "mean_discovery_messages", "(b) Communication", "Messages/request"),
        (axes[2], "p95_success_latency_ms", "(c) Tail latency", "Simulated P95 (ms)"),
    ):
        values = []
        lows = []
        highs = []
        colors = []
        for key in order:
            row = data[data.pipeline.eq(key)].iloc[0]
            value = float(row[field])
            values.append(value)
            lows.append(value - float(row[f"{field}_ci_low"]))
            highs.append(float(row[f"{field}_ci_high"]) - value)
            colors.append(COLORS[key])
        axes_bar = ax.bar(x, values, color=colors, width=0.68, alpha=0.88)
        ax.errorbar(x, values, yerr=np.vstack([lows, highs]), fmt="none", color="#222222", capsize=2)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [PROTOCOL_LABELS[key] for key in order], rotation=28, ha="right")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    axes[0].set_ylim(0, 105)
    export_figure(fig, output_dir, "decentralized_protocol_main")
    plt.close(fig)


def plot_robustness(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    panels = (
        ("outage_rate", "Node outage (%)"),
        ("topology_churn_rate", "Changed business links (%)"),
        ("resource_change_probability", "Nodes with resource drop (%)"),
        ("malicious_rate", "Malicious nodes (%)"),
    )
    for ax, (factor, xlabel) in zip(axes.flat, panels):
        if factor == "malicious_rate":
            data = summary[summary.experiment.eq("malicious_protocol")].copy()
            data["x_value"] = 100.0 * data.malicious_rate
        else:
            data = summary[
                summary.experiment.eq("dynamic_robustness") & summary.factor.eq(factor)
            ].copy()
            data["x_value"] = 100.0 * data.level
        for key in DECENTRALIZED_LABELS:
            subset = data[data.pipeline.eq(key)]
            if len(subset):
                errorbar(ax, subset, "x_value", "honest_success_percent", key, PROTOCOL_LABELS[key])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Honest candidate discovery (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6)
        ax.set_title(
            {
                "outage_rate": "(a) Node outages",
                "topology_churn_rate": "(b) Stale topology",
                "resource_change_probability": "(c) In-search resource drops",
                "malicious_rate": "(d) Malicious candidates",
            }[factor],
            loc="left",
            fontweight="bold",
        )
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    export_figure(fig, output_dir, "decentralized_search_robustness")
    plt.close(fig)


def plot_trust(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[summary.experiment.eq("trust_confirmation_ablation")].copy()
    data["malicious_percent"] = 100.0 * data.malicious_rate
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    for key, label in TRUST_LABELS.items():
        subset = data[data.pipeline.eq(key)]
        errorbar(axes[0], subset, "malicious_percent", "honest_success_percent", key, label)
        errorbar(axes[1], subset, "malicious_percent", "malicious_selection_percent", key, label)
    axes[0].set_title("(a) Honest candidate discovery", loc="left", fontweight="bold")
    axes[0].set_ylabel("Success (%)")
    axes[1].set_title("(b) Malicious candidate selection", loc="left", fontweight="bold")
    axes[1].set_ylabel("Selection (%)")
    for ax in axes:
        ax.set_xlabel("Malicious nodes (%)")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5)
    export_figure(fig, output_dir, "trust_confirmation_ablation")
    plt.close(fig)


def plot_hyperparameters(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[summary.experiment.eq("hyperparameter")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    labels = {
        "max_hops": ("Maximum forwarding depth", "(a) Forwarding depth"),
        "deadline_ms": ("Deadline (ms)", "(b) Response deadline"),
        "top_k": ("Confirmed candidates (top-k)", "(c) Confirmation breadth"),
        "gossip_fanout": ("Gossip fanout", "(d) Gossip fanout"),
    }
    for ax, factor in zip(axes.flat, HYPER_LEVELS):
        subset = data[data.factor.eq(factor)].sort_values("level")
        color = COLORS["gossip"] if factor == "gossip_fanout" else COLORS["proposed"]
        ax.plot(subset.level, subset.honest_success_percent, color=color, marker="o", linewidth=1.4)
        ax.fill_between(
            subset.level,
            subset.honest_success_percent_ci_low,
            subset.honest_success_percent_ci_high,
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        ax.axvline(REFERENCE_LEVEL[factor], color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xlabel(labels[factor][0])
        ax.set_ylabel("Honest discovery (%)")
        ax.set_title(labels[factor][1], loc="left", fontweight="bold")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    export_figure(fig, output_dir, "search_hyperparameter_sensitivity")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    p_value_columns = {"wilcoxon_p", "holm_adjusted_p"}
    for values in frame[list(columns)].itertuples(index=False, name=None):
        formatted = []
        for column, value in zip(columns, values):
            if pd.isna(value):
                formatted.append("NA")
            elif isinstance(value, (float, np.floating)):
                numeric = float(value)
                if column in p_value_columns and 0.0 < numeric < 1e-3:
                    formatted.append(f"{numeric:.3e}")
                else:
                    formatted.append(f"{numeric:.4f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def validate_results(
    raw: pd.DataFrame,
    network: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: FullAblationConfig,
    sections: set[str],
) -> dict[str, object]:
    """Fail closed on structural, accounting, and method-unification errors."""

    method_conditions = 0
    if "protocol" in sections:
        method_conditions += len(PROTOCOL_LABELS)
        method_conditions += 4 * len(DECENTRALIZED_LABELS)
    if "robustness" in sections:
        method_conditions += sum(len(levels) for levels in ROBUSTNESS_LEVELS.values()) * 3
    if "trust" in sections:
        method_conditions += 4 * len(TRUST_LABELS)
    if "hyperparameter" in sections:
        method_conditions += sum(len(levels) for levels in HYPER_LEVELS.values())

    expected_raw = method_conditions * cfg.network_repeats * cfg.tasks_per_network
    expected_network = method_conditions * cfg.network_repeats
    key = [
        "experiment",
        "factor",
        "level",
        "malicious_rate",
        "pipeline",
        "repeat",
        "task",
    ]
    message_columns = [
        "search_messages",
        "request_messages",
        "response_messages",
        "confirmation_messages",
        "total_messages",
    ]
    checks = {
        "raw_rows_match": len(raw) == expected_raw,
        "network_rows_match": len(network) == expected_network,
        "summary_groups_match": len(summary) == method_conditions,
        "no_duplicate_task_keys": not raw.duplicated(key).any(),
        "all_success_values_binary": bool(
            raw[["success", "honest_success", "malicious_selected"]]
            .isin([0, 1])
            .all()
            .all()
        ),
        "honest_success_implies_success": bool((raw.honest_success <= raw.success).all()),
        "nonnegative_message_counts": bool((raw[message_columns] >= 0).all().all()),
        "search_message_accounting_exact": bool(
            np.allclose(
                raw.search_messages,
                raw.request_messages + raw.response_messages + raw.confirmation_messages,
            )
        ),
        "finite_cache_and_update_metrics": bool(
            np.isfinite(raw[["cache_wrmse", "update_rate"]].to_numpy()).all()
        ),
        "single_shared_cache_replay": raw.cache_wrmse.nunique() == 1,
        "update_budget_exact": bool(np.allclose(raw.update_rate, cfg.update_budget)),
        "all_replay_keys_are_original_mpc": all(
            spec[0] == "runtime_mpc_dp"
            for spec in [*PROTOCOL_SPEC.values(), *TRUST_SPEC.values()]
        ),
        "no_threshold_aware_penalty": cfg.threshold_lambda == 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"formal result validation failed: {failed}")
    return {
        "status": "PASS",
        "checks": checks,
        "expected_raw_rows": expected_raw,
        "actual_raw_rows": len(raw),
        "expected_network_rows": expected_network,
        "actual_network_rows": len(network),
        "summary_groups": len(summary),
        "independent_networks_per_condition": cfg.network_repeats,
        "tasks_per_network": cfg.tasks_per_network,
        "update_rate": float(raw.update_rate.iloc[0]),
        "cache_wrmse": float(raw.cache_wrmse.iloc[0]),
    }


def write_report(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    hyper_tests: pd.DataFrame,
    cfg: FullAblationConfig,
    output_dir: Path,
) -> None:
    protocol = summary[summary.experiment.eq("protocol_main")].copy()
    protocol["method"] = protocol.pipeline.map(PROTOCOL_LABELS)
    dynamic = summary[summary.experiment.eq("dynamic_robustness")].copy()
    malicious = summary[summary.experiment.eq("malicious_protocol")].copy()
    trust = summary[summary.experiment.eq("trust_confirmation_ablation")].copy()
    trust["method"] = trust.pipeline.map(TRUST_LABELS)
    hyper = summary[summary.experiment.eq("hyperparameter")].copy()

    report = f"""# 工控机100节点去中心化候选发现统一仿真实验

## 1. 结论边界

本实验在工控机上执行真实代码，但网络时延来自离散事件模型，因此报告的是**仿真候选发现时延**，不是100台物理节点的实测网络时延。实验终点为卸载前可信候选输出，不执行任务迁移与运行。

## 2. 冻结协议

- 节点数：{cfg.num_nodes}；平均业务度数：{cfg.average_degree}。
- 每个条件：{cfg.network_repeats}个独立网络，每个网络{cfg.tasks_per_network}个连续远端必需任务；独立统计单位是网络。
- 资源状态：真实轨迹驱动的FITS-Delta预测与论文原始MPC-DP缓存，更新预算{cfg.update_budget:.0%}，决策参数$\\eta={cfg.eta:g}$；不使用门槛感知惩罚项。
- 默认动态条件：节点断联{cfg.outage_rate:.0%}、业务链路变化{cfg.topology_churn_rate:.0%}、搜索期资源下降节点{cfg.resource_change_probability:.0%}。
- 搜索：最大{cfg.max_hops}跳，候选收集{cfg.candidate_collection_ms:g} ms，绝对截止{cfg.deadline_ms:g} ms，Top-{cfg.top_k}并行确认。
- Random Walk：单令牌、最多{cfg.random_walk_restarts}次顺序起点重启，不发生分支。
- Expanding Ring：TTL从1递增到{cfg.max_hops}，每{cfg.expanding_ring_interval_ms:g} ms启动下一层业务关系环；不同环的重复流量计入消息开销。
- 所有去中心化方法只沿业务关系边传播；Oracle具有全局真实状态，仅作为不可部署上界。

## 3. 去中心化方法主比较（0%恶意节点）

{markdown_table(protocol, ['method', 'networks', 'honest_success_percent', 'honest_success_percent_ci_low', 'honest_success_percent_ci_high', 'median_success_latency_ms', 'p95_success_latency_ms', 'mean_discovery_messages', 'mean_request_messages', 'mean_response_messages', 'mean_confirmation_messages', 'mean_duplicate_requests', 'mean_selected_hops'])}

## 4. 动态鲁棒性

{markdown_table(dynamic.assign(method=dynamic.pipeline.map(PROTOCOL_LABELS)), ['factor', 'level', 'method', 'honest_success_percent', 'honest_success_percent_ci_low', 'honest_success_percent_ci_high', 'p95_success_latency_ms', 'mean_discovery_messages'])}

## 5. 恶意节点鲁棒性

{markdown_table(malicious.assign(method=malicious.pipeline.map(PROTOCOL_LABELS)), ['malicious_rate', 'method', 'honest_success_percent', 'honest_success_percent_ci_low', 'honest_success_percent_ci_high', 'malicious_selection_percent', 'p95_success_latency_ms', 'mean_discovery_messages'])}

## 6. 信任与二次确认消融

{markdown_table(trust, ['malicious_rate', 'method', 'honest_success_percent', 'honest_success_percent_ci_low', 'honest_success_percent_ci_high', 'malicious_selection_percent', 'mean_confirmations', 'mean_invalid_confirmations', 'p95_success_latency_ms', 'mean_discovery_messages'])}

## 7. 超参数敏感性

{markdown_table(hyper, ['factor', 'level', 'pipeline', 'malicious_rate', 'honest_success_percent', 'honest_success_percent_ci_low', 'honest_success_percent_ci_high', 'p95_success_latency_ms', 'mean_discovery_messages', 'mean_confirmations'])}

## 8. 网络级配对检验

{markdown_table(tests, ['experiment', 'factor', 'level', 'malicious_rate', 'endpoint', 'comparison', 'network_clusters', 'proposed_minus_baseline', 'cluster_bootstrap_ci_low', 'cluster_bootstrap_ci_high', 'wilcoxon_p', 'holm_adjusted_p'])}

## 9. 超参数相对默认值检验

{markdown_table(hyper_tests, ['factor', 'reference_level', 'alternative_level', 'endpoint', 'network_clusters', 'alternative_minus_reference', 'cluster_bootstrap_ci_low', 'cluster_bootstrap_ci_high', 'wilcoxon_p', 'holm_adjusted_p'])}

## 10. 解释规则

1. 方法主比较改变的只是查询传播方式；FITS-Delta、论文原始MPC-DP、缓存、信任排序和二次确认均相同。
2. 信任消融固定Workflow传播，避免把信任收益与覆盖范围混淆。
3. 成功率只统计在截止时间内返回且确认有效的诚实候选；P95只在成功请求上计算，必须与成功率一起解释。
4. 消息数分为请求、候选响应和确认消息；状态更新消息不混入搜索层通信开销。
5. 统计推断在网络级进行，报告效应量、聚类bootstrap 95%区间、配对Wilcoxon检验和Holm校正结果。
6. Centralized Oracle的通信与时延不可与去中心化方法作公平部署比较，仅提供成功率上界。
7. 本实验不用于声称某方法在所有指标全面占优；应报告成功率、时延和通信之间的权衡。
"""
    (output_dir / "工控机100节点去中心化候选发现统一仿真实验结果与分析.md").write_text(
        report, encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "industrial_pc_100node_discovery_20260910",
    )
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--tasks-per-network", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--trace-length",
        type=int,
        default=282,
        help="must be divisible by the six-slot deployed update-budget epoch",
    )
    parser.add_argument(
        "--sections",
        default="protocol,robustness,trust,hyperparameter",
        help="comma-separated subset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trace_length % 6:
        raise ValueError(
            "trace length must be divisible by the six-slot deployed update-budget epoch"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = base_config(args.repeats, args.tasks_per_network, args.trace_length)
    nodes_actual, replays = build_inputs(args.data_dir, cfg)
    sections = {item.strip() for item in args.sections.split(",") if item.strip()}
    raw_frames: list[pd.DataFrame] = []

    if "protocol" in sections:
        raw_frames.append(
            run_block(
                args.output_dir,
                "protocol_main",
                "protocol_main",
                "malicious_rate",
                0.0,
                0.0,
                cfg,
                PROTOCOL_LABELS,
                PROTOCOL_SPEC,
                nodes_actual,
                replays,
                args.workers,
            )
        )
        for rate in (0.0, 0.10, 0.20, 0.40):
            raw_frames.append(
                run_block(
                    args.output_dir,
                    f"malicious_protocol_{int(100*rate):02d}",
                    "malicious_protocol",
                    "malicious_rate",
                    rate,
                    rate,
                    cfg,
                    DECENTRALIZED_LABELS,
                    DECENTRALIZED_SPEC,
                    nodes_actual,
                    replays,
                    args.workers,
                )
            )

    if "robustness" in sections:
        robust_labels = {
            key: DECENTRALIZED_LABELS[key]
            for key in ("proposed", "gossip", "expanding_ring")
        }
        robust_spec = {key: DECENTRALIZED_SPEC[key] for key in robust_labels}
        for factor, levels in ROBUSTNESS_LEVELS.items():
            for level in levels:
                changes = {factor: float(level)}
                if factor == "resource_change_probability":
                    changes["resource_shock_std"] = 11.0 if level > 0 else 0.0
                scenario_cfg = replace(cfg, **changes)
                raw_frames.append(
                    run_block(
                        args.output_dir,
                        f"robust_{factor}_{int(100*level):02d}",
                        "dynamic_robustness",
                        factor,
                        level,
                        0.0,
                        scenario_cfg,
                        robust_labels,
                        robust_spec,
                        nodes_actual,
                        replays,
                        args.workers,
                    )
                )

    if "trust" in sections:
        for rate in (0.0, 0.10, 0.20, 0.40):
            raw_frames.append(
                run_block(
                    args.output_dir,
                    f"trust_confirmation_{int(100*rate):02d}",
                    "trust_confirmation_ablation",
                    "malicious_rate",
                    rate,
                    rate,
                    cfg,
                    TRUST_LABELS,
                    TRUST_SPEC,
                    nodes_actual,
                    replays,
                    args.workers,
                )
            )

    if "hyperparameter" in sections:
        for factor, levels in HYPER_LEVELS.items():
            for level in levels:
                changes: dict[str, int | float] = {}
                malicious_rate = 0.20 if factor == "top_k" else 0.0
                if factor == "max_hops":
                    changes["max_hops"] = int(level)
                elif factor == "deadline_ms":
                    changes["deadline_ms"] = float(level)
                    changes["candidate_collection_ms"] = 2.0 * float(level) / 3.0
                    changes["expanding_ring_interval_ms"] = (
                        changes["candidate_collection_ms"] / cfg.max_hops
                    )
                elif factor == "top_k":
                    changes["top_k"] = int(level)
                elif factor == "gossip_fanout":
                    changes["gossip_fanout"] = int(level)
                scenario_cfg = replace(cfg, **changes)
                if factor == "gossip_fanout":
                    labels = {"gossip": PROTOCOL_LABELS["gossip"]}
                    spec = {"gossip": PROTOCOL_SPEC["gossip"]}
                else:
                    labels = {"proposed": PROTOCOL_LABELS["proposed"]}
                    spec = {"proposed": PROTOCOL_SPEC["proposed"]}
                raw_frames.append(
                    run_block(
                        args.output_dir,
                        f"hyper_{factor}_{int(level):03d}",
                        "hyperparameter",
                        factor,
                        level,
                        malicious_rate,
                        scenario_cfg,
                        labels,
                        spec,
                        nodes_actual,
                        replays,
                        args.workers,
                    )
                )

    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(args.output_dir / "industrial_100node_discovery_raw.csv", index=False)
    network = network_summary(raw)
    network.to_csv(args.output_dir / "industrial_100node_discovery_network.csv", index=False)
    summary = summarize(network)
    summary.to_csv(args.output_dir / "industrial_100node_discovery_summary.csv", index=False)

    tests = pd.concat(
        [
            paired_comparison(
                network,
                "protocol_main",
                "proposed",
                ("gossip", "random_walk", "expanding_ring"),
                ("honest_success_rate", "mean_discovery_messages"),
                910_000,
            ),
            paired_comparison(
                network,
                "dynamic_robustness",
                "proposed",
                ("gossip", "expanding_ring"),
                ("honest_success_rate", "mean_discovery_messages"),
                920_000,
            ),
            paired_comparison(
                network,
                "malicious_protocol",
                "proposed",
                ("gossip", "random_walk", "expanding_ring"),
                ("honest_success_rate", "malicious_selection_rate"),
                930_000,
            ),
            paired_comparison(
                network,
                "trust_confirmation_ablation",
                "path_trust",
                ("direct_only", "no_trust", "no_confirmation", "oracle_trust"),
                ("honest_success_rate", "malicious_selection_rate"),
                940_000,
            ),
        ],
        ignore_index=True,
    )
    tests.to_csv(args.output_dir / "industrial_100node_discovery_paired_tests.csv", index=False)
    hyper_tests = hyperparameter_tests(network)
    hyper_tests.to_csv(args.output_dir / "industrial_100node_hyperparameter_tests.csv", index=False)

    validation = validate_results(raw, network, summary, cfg, sections)
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    configure_plotting()
    plot_protocol(summary, args.output_dir)
    plot_robustness(summary, args.output_dir)
    plot_trust(summary, args.output_dir)
    plot_hyperparameters(summary, args.output_dir)
    write_report(summary, tests, hyper_tests, cfg, args.output_dir)

    source_files = [
        Path(__file__),
        RAFT_DIR / "edgebench_core" / "update_policies.py",
        RAFT_DIR / "joint_experiments" / "run_deployment_equivalent_network_scale.py",
        RAFT_DIR / "joint_experiments" / "run_trust_robustness_experiment.py",
        RAFT_DIR / "joint_experiments" / "run_full_pipeline_ablation.py",
        RAFT_DIR / "joint_experiments" / "run_joint_discovery_experiment.py",
        RAFT_DIR / "search_experiments" / "run_dynamic_search_experiments.py",
        RAFT_DIR / "update_experiments" / "run_update_experiments.py",
    ]
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "prediction_model": "FITS-Delta",
        "data_dir": str(args.data_dir.resolve()),
        "input_sha256": {
            name: sha256_file(args.data_dir / name)
            for name in ("value.csv", "cpu_predictions.csv", "memory_predictions.csv")
        },
        "config": asdict(cfg),
        "update_profile": {
            "implementation": "edgebench_core.runtime_update_proposal",
            "strategy": "mpc_dp",
            "budget_guard": "RepeatedEpochBudgetGuard",
            "epoch_slots": cfg.horizon,
            "budget": cfg.update_budget,
            "eta": cfg.eta,
            "threshold_aware_extension": False,
            "threshold_lambda": None,
            "observed_update_rate": replays["runtime_mpc_dp"].update_rate,
            "cache_wrmse": replays["runtime_mpc_dp"].cache_wrmse,
        },
        "sections": sorted(sections),
        "method_labels": PROTOCOL_LABELS,
        "method_specifications": PROTOCOL_SPEC,
        "trust_labels": TRUST_LABELS,
        "trust_specifications": TRUST_SPEC,
        "independent_unit": "generated business network",
        "within_network_repeated_observations": cfg.tasks_per_network,
        "latency_type": "discrete-event simulated candidate-discovery latency",
        "source_sha256": {
            str(path.relative_to(RAFT_DIR)): sha256_file(path) for path in source_files
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Results saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
