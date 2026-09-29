from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_cluster_and_gossip_budget_analysis import bootstrap_mean_ci
from run_joint_discovery_experiment import (
    BudgetGuard,
    CacheReplay,
    JointConfig,
    build_cache_replays,
    build_node_replays,
    load_inputs,
    markdown_table,
    run_joint_trials,
    summarize,
    weighted_squared_error,
)


HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent

PIPELINE_LABELS = {
    "threshold_mpc": "Threshold-aware FITS + MPC-DP",
    "proposed": "Original FITS + MPC-DP",
    "last_mpc": "Last-value + MPC-DP",
    "fits_periodic": "FITS + Periodic",
    "fits_delta": "FITS + Send-on-delta",
    "oracle_state": "Fresh-state oracle",
}

COLORS = {
    "threshold_mpc": "#0072B2",
    "proposed": "#56B4E9",
    "last_mpc": "#CC79A7",
    "fits_periodic": "#E69F00",
    "fits_delta": "#D55E00",
    "oracle_state": "#666666",
}


def export_threshold_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Export one editable-vector and two review-ready raster formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def build_validation_thresholds(calibration_actual: np.ndarray) -> np.ndarray:
    """Create fixed CPU/memory demand profiles using calibration-only observations."""
    availability = 100.0 - np.asarray(calibration_actual, dtype=float)
    quantiles = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])
    cpu_demands = 0.95 * np.quantile(availability[:, 0], quantiles)
    memory_demands = 0.95 * np.quantile(availability[:, 1], quantiles)
    demand_grid = np.asarray(
        [(cpu, memory) for cpu in cpu_demands for memory in memory_demands],
        dtype=float,
    )
    # Eligibility in utilization space is utilization <= 100 - demand.
    return 100.0 - demand_grid


def threshold_classification_cost(
    state: np.ndarray,
    cached: np.ndarray,
    utilization_thresholds: np.ndarray,
    false_negative_weight: float = 2.0,
) -> float:
    true_eligible = np.all(state <= utilization_thresholds, axis=1)
    cached_eligible = np.all(cached <= utilization_thresholds, axis=1)
    false_negative = true_eligible & ~cached_eligible
    false_positive = ~true_eligible & cached_eligible
    return float(
        np.mean(false_negative_weight * false_negative.astype(float) + false_positive.astype(float))
    )


def threshold_mpc_first_action(
    forecast: np.ndarray,
    cached: np.ndarray,
    max_updates: int,
    eta: float,
    threshold_lambda: float,
    utilization_thresholds: np.ndarray,
    resource_weights: np.ndarray | None = None,
) -> bool:
    length = int(forecast.shape[0])
    max_updates = min(max(int(max_updates), 0), length)

    @lru_cache(maxsize=None)
    def solve(h: int, used: int, last_index: int) -> tuple[float, tuple[int, ...]]:
        if h == length:
            return (0.0, ()) if used == max_updates else (float("inf"), ())
        cache_value = cached if last_index < 0 else forecast[last_index]
        slots_after_current = length - h - 1
        updates_needed = max_updates - used
        best = (float("inf"), ())
        if updates_needed <= slots_after_current:
            future_cost, future_plan = solve(h + 1, used, last_index)
            continuous_cost = weighted_squared_error(
                forecast[h], cache_value, resource_weights
            )
            threshold_cost = threshold_classification_cost(
                forecast[h], cache_value, utilization_thresholds
            )
            waiting_penalty = float(eta) if used > 0 else 0.0
            total = (
                continuous_cost
                + float(threshold_lambda) * threshold_cost
                + waiting_penalty
                + future_cost
            )
            best = (total, (0,) + future_plan)
        if used < max_updates:
            send_cost, send_plan = solve(h + 1, used + 1, h)
            candidate = (send_cost, (1,) + send_plan)
            if candidate[0] < best[0] - 1e-12:
                best = candidate
        return best

    return bool(solve(0, 0, -1)[1][0])


def replay_threshold_strategy(
    actual: np.ndarray,
    predictions: np.ndarray,
    initial_cache: np.ndarray,
    budget: float,
    eta: float,
    threshold_lambda: float,
    utilization_thresholds: np.ndarray,
    horizon: int = 6,
    resource_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cache = np.asarray(initial_cache, dtype=float).copy()
    cache_history = np.empty_like(actual)
    actions = np.zeros(len(actual), dtype=bool)
    guard = BudgetGuard(len(actual), budget)
    for t in range(len(actual)):
        forecast = predictions[t, : min(horizon, predictions.shape[1])]
        local_budget = min(
            guard.target - guard.used,
            int(math.ceil(budget * len(forecast))),
        )
        proposed = threshold_mpc_first_action(
            forecast,
            cache,
            local_budget,
            eta,
            threshold_lambda,
            utilization_thresholds,
            resource_weights,
        )
        action = guard.apply(proposed, t)
        actions[t] = action
        if action:
            cache = actual[t].copy()
        cache_history[t] = cache
    if int(actions.sum()) != guard.target:
        raise AssertionError(f"threshold MPC used {actions.sum()} updates, expected {guard.target}")
    return cache_history, actions


def threshold_confusion(
    actual: np.ndarray,
    cache: np.ndarray,
    utilization_thresholds: np.ndarray,
) -> tuple[float, float, float]:
    actual_eligible = np.all(
        actual[:, None, :] <= utilization_thresholds[None, :, :], axis=2
    )
    cached_eligible = np.all(
        cache[:, None, :] <= utilization_thresholds[None, :, :], axis=2
    )
    tp = int(np.sum(actual_eligible & cached_eligible))
    fn = int(np.sum(actual_eligible & ~cached_eligible))
    fp = int(np.sum(~actual_eligible & cached_eligible))
    fnr = fn / (tp + fn) if tp + fn else 0.0
    fdr = fp / (tp + fp) if tp + fp else 0.0
    wrmse = float(np.sqrt(np.mean(np.sum(0.5 * np.square(actual - cache), axis=1))))
    return fnr, fdr, wrmse


def calibrate_parameters(
    actual: np.ndarray,
    predictions: np.ndarray,
    split: int,
    budget: float,
    horizon: int,
) -> tuple[float, float, np.ndarray, pd.DataFrame]:
    calibration_end = split // 2
    thresholds = build_validation_thresholds(actual[:calibration_end])
    validation_start = calibration_end
    validation_end = split - horizon
    validation_actual = actual[validation_start:validation_end]
    validation_predictions = predictions[validation_start:validation_end]
    validation_initial = actual[validation_start - 1]
    eta_grid = (0.0, 0.1, 0.5, 1.0, 2.0)
    lambda_grid = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0)
    rows = []
    for eta in eta_grid:
        for threshold_lambda in lambda_grid:
            cache, actions = replay_threshold_strategy(
                validation_actual,
                validation_predictions,
                validation_initial,
                budget,
                eta,
                threshold_lambda,
                thresholds,
                horizon,
            )
            fnr, fdr, wrmse = threshold_confusion(
                validation_actual, cache, thresholds
            )
            # False exclusions are twice as costly because they directly remove
            # feasible discovery candidates. WRMSE is only a deterministic tie-breaker.
            selection_score = 2.0 * fnr + fdr
            rows.append(
                {
                    "eta": eta,
                    "threshold_lambda": threshold_lambda,
                    "validation_fnr": fnr,
                    "validation_false_discovery_rate": fdr,
                    "validation_wrmse": wrmse,
                    "selection_score": selection_score,
                    "update_count": int(actions.sum()),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["selection_score", "validation_wrmse", "eta", "threshold_lambda"]
    ).reset_index(drop=True)
    best = frame.iloc[0]
    return float(best.eta), float(best.threshold_lambda), thresholds, frame


def add_threshold_replay(
    nodes_actual: np.ndarray,
    nodes_predictions: np.ndarray,
    replays: dict[str, CacheReplay],
    cfg: JointConfig,
    eta: float,
    threshold_lambda: float,
    thresholds: np.ndarray,
    resource_weights: np.ndarray | None = None,
) -> dict[str, CacheReplay]:
    caches = []
    actions = []
    for node in range(cfg.num_nodes):
        cache, action = replay_threshold_strategy(
            nodes_actual[node],
            nodes_predictions[node],
            nodes_actual[node, 0],
            cfg.update_budget,
            eta,
            threshold_lambda,
            thresholds,
            cfg.horizon,
            resource_weights,
        )
        caches.append(cache)
        actions.append(action)
    cache_array = np.asarray(caches)
    action_array = np.asarray(actions)
    errors = nodes_actual - cache_array
    wrmse = float(np.sqrt(np.mean(np.sum(0.5 * np.square(errors), axis=2))))
    threshold_replay = CacheReplay(
        pipeline="threshold_mpc",
        cache_utilization=cache_array,
        actions=action_array,
        cache_wrmse=wrmse,
        update_count=int(action_array.sum()),
        update_rate=float(action_array.mean()),
    )
    return {"threshold_mpc": threshold_replay, **replays}


def network_cluster_tests(raw: pd.DataFrame) -> pd.DataFrame:
    comparators = ["proposed", "last_mpc", "fits_periodic", "fits_delta"]
    rows = []
    for workload_idx, workload in enumerate(raw.workload.unique()):
        rates = (
            raw[raw.workload == workload]
            .groupby(["repeat", "pipeline"], as_index=False)
            .success.mean()
        )
        proposed = rates[rates.pipeline == "threshold_mpc"][["repeat", "success"]].rename(
            columns={"success": "threshold_mpc"}
        )
        for comparator_idx, comparator in enumerate(comparators):
            baseline = rates[rates.pipeline == comparator][["repeat", "success"]].rename(
                columns={"success": "baseline"}
            )
            pair = proposed.merge(baseline, on="repeat", validate="one_to_one")
            difference = 100.0 * (pair.threshold_mpc - pair.baseline).to_numpy()
            ci_low, ci_high = bootstrap_mean_ci(
                difference,
                seed=120_000 + workload_idx * 100 + comparator_idx,
            )
            p_value = (
                1.0
                if np.allclose(difference, 0.0)
                else float(
                    wilcoxon(
                        difference,
                        zero_method="pratt",
                        alternative="two-sided",
                        method="auto",
                    ).pvalue
                )
            )
            rows.append(
                {
                    "workload": workload,
                    "comparison": f"threshold_mpc vs {comparator}",
                    "network_clusters": len(pair),
                    "threshold_mpc_mean_percent": 100.0 * pair.threshold_mpc.mean(),
                    "baseline_mean_percent": 100.0 * pair.baseline.mean(),
                    "paired_difference_pp": difference.mean(),
                    "cluster_bootstrap_ci_low_pp": ci_low,
                    "cluster_bootstrap_ci_high_pp": ci_high,
                    "wilcoxon_network_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def plot_results(
    calibration: pd.DataFrame,
    summary: pd.DataFrame,
    cluster_tests: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    order = [
        "threshold_mpc",
        "proposed",
        "last_mpc",
        "fits_periodic",
        "fits_delta",
        "oracle_state",
    ]
    labels = ["Threshold", "Original", "Last", "Periodic", "Delta", "Oracle"]
    colors = [COLORS[x] for x in order]
    remote = summary[summary.workload == "remote_required"].set_index("pipeline").loc[order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), constrained_layout=True)

    pivot = calibration.pivot(index="eta", columns="threshold_lambda", values="selection_score")
    image = axes[0, 0].imshow(pivot.to_numpy(), aspect="auto", cmap="Blues_r")
    axes[0, 0].set_xticks(np.arange(len(pivot.columns)))
    axes[0, 0].set_xticklabels([f"{x:g}" for x in pivot.columns])
    axes[0, 0].set_yticks(np.arange(len(pivot.index)))
    axes[0, 0].set_yticklabels([f"{x:g}" for x in pivot.index])
    axes[0, 0].set_xlabel("Threshold-loss weight, lambda")
    axes[0, 0].set_ylabel("MPC waiting parameter, eta")
    axes[0, 0].set_title("(a) Validation-only calibration", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[0, 0], label="2×FNR + false-discovery rate")

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
    axes[0, 1].set_ylabel("Remote candidate discovery (%)")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_title("(b) Held-out discovery", loc="left", fontweight="bold")

    width = 0.36
    axes[1, 0].bar(
        x - width / 2,
        100.0 * remote.candidate_micro_recall,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        label="Candidate recall",
    )
    axes[1, 0].bar(
        x + width / 2,
        100.0 * remote.candidate_micro_precision,
        width,
        color="white",
        edgecolor=colors,
        linewidth=1.0,
        label="Candidate precision",
    )
    axes[1, 0].set_ylabel("Remote resource classification (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].set_title("(c) Candidate-level validity", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=6.2, loc="lower right")

    axes[1, 1].scatter(
        remote.cache_wrmse,
        remote.success_percent,
        c=colors,
        edgecolor="black",
        linewidth=0.4,
        s=40,
    )
    for idx, label in enumerate(labels):
        axes[1, 1].annotate(
            label,
            (remote.cache_wrmse.iloc[idx], remote.success_percent.iloc[idx]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
        )
    axes[1, 1].set_xlabel("State-cache WRMSE")
    axes[1, 1].set_ylabel("Remote candidate discovery (%)")
    axes[1, 1].set_title("(d) Error–discovery mismatch", loc="left", fontweight="bold")

    for ax in axes.ravel()[1:3]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
    for ax in axes.ravel():
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    export_threshold_figure(fig, output_dir, "threshold_aware_mpc_evaluation")
    plt.close(fig)


def write_report(
    calibration: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    selected_eta: float,
    selected_lambda: float,
    split: int,
    cfg: JointConfig,
    output_dir: Path,
) -> None:
    display = summary.copy()
    display["pipeline"] = display.pipeline.map(PIPELINE_LABELS)
    remote = summary[summary.workload == "remote_required"].set_index("pipeline")
    threshold = remote.loc["threshold_mpc"]
    original = remote.loc["proposed"]
    periodic = remote.loc["fits_periodic"]
    report = f"""# 门槛感知MPC-DP候选发现实验

## 1. 无测试泄漏的设计

- 原始轨迹前{split}个时隙不进入正式测试：前半段仅构造25个CPU/内存资源门槛，后半段仅选择eta和lambda；最后{len(pd.read_csv(RAFT_DIR / 'test' / 'value.csv', header=None)) - split}个时隙为冻结参数后的测试区间。
- 门槛来自校准段可用资源的10%、25%、50%、75%、90%分位数组合，并乘以固定0.95需求比例；没有读取测试任务需求。
- 验证选择目标为2×候选假阴性率+候选假发现率，假阴性权重更高是因为它会直接删除真实可用节点；WRMSE仅用于分数相同时的确定性排序。
- 选定参数：eta={selected_eta:g}，lambda={selected_lambda:g}。正式测试中所有可部署方法仍严格使用{cfg.update_budget:.0%}更新预算。

## 2. 验证参数表

{markdown_table(calibration, ['eta', 'threshold_lambda', 'validation_fnr', 'validation_false_discovery_rate', 'validation_wrmse', 'selection_score', 'update_count'])}

## 3. 独立测试候选发现结果

{markdown_table(display, ['workload', 'pipeline', 'success_percent', 'success_ci95_percent', 'cache_wrmse', 'candidate_micro_recall', 'candidate_micro_precision', 'cache_blocked_discovery_percent', 'protocol_failure_with_candidate_percent', 'p95_success_latency_ms', 'mean_search_messages', 'mean_update_messages'])}

## 4. 网络级配对统计

{markdown_table(tests, ['workload', 'comparison', 'network_clusters', 'threshold_mpc_mean_percent', 'baseline_mean_percent', 'paired_difference_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_network_p'])}

## 5. 关键解释

- 远端场景中门槛感知MPC-DP的发现率为{threshold.success_percent:.2f}%，原MPC-DP为{original.success_percent:.2f}%，周期更新为{periodic.success_percent:.2f}%。
- 门槛感知方法的候选召回率为{100 * threshold.candidate_micro_recall:.2f}%，缓存阻断率为{threshold.cache_blocked_discovery_percent:.2f}%；原MPC-DP分别为{100 * original.candidate_micro_recall:.2f}%和{original.cache_blocked_discovery_percent:.2f}%。
- 如果门槛感知方法只改善验证指标而没有改善独立测试发现率，应视为该代价设计无效，不能替换主方法；如果改善但仍低于Periodic，则应保留完整比较并限制优势表述。

## 6. 边界

1. 门槛集合表示校准期常见资源需求等级，不等同于提前知道某个测试任务。
2. 10个节点仍来自同一Alibaba轨迹的错位回放，结果是候选发现仿真而非物理部署。
3. 参数搜索空间、验证目标和所有失败结果均完整保存，不能只报告表现最好的测试配置。
"""
    (output_dir / "门槛感知MPC-DP实验结果与分析.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "threshold_mpc_formal",
    )
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--tasks-per-network", type=int, default=120)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = JointConfig(network_repeats=args.repeats, tasks_per_network=args.tasks_per_network)
    actual, predictions = load_inputs(args.data_dir)
    split = cfg.trace_start
    if abs(split / len(actual) - args.validation_ratio) > 0.01:
        raise ValueError(
            f"Configured trace start {split} is inconsistent with validation ratio {args.validation_ratio}"
        )
    selected_eta, selected_lambda, thresholds, calibration = calibrate_parameters(
        actual,
        predictions,
        split,
        cfg.update_budget,
        cfg.horizon,
    )
    nodes_actual, nodes_predictions = build_node_replays(actual, predictions, cfg)
    replays = build_cache_replays(nodes_actual, nodes_predictions, cfg)
    replays = add_threshold_replay(
        nodes_actual,
        nodes_predictions,
        replays,
        cfg,
        selected_eta,
        selected_lambda,
        thresholds,
    )
    pipelines = [
        "threshold_mpc",
        "proposed",
        "last_mpc",
        "fits_periodic",
        "fits_delta",
        "oracle_state",
    ]
    raw = run_joint_trials(nodes_actual, replays, cfg, pipelines=pipelines)
    summary = summarize(raw)
    tests = network_cluster_tests(raw)

    calibration.to_csv(output_dir / "threshold_mpc_calibration.csv", index=False)
    pd.DataFrame(thresholds, columns=["cpu_utilization_threshold", "memory_utilization_threshold"]).to_csv(
        output_dir / "calibration_thresholds.csv", index=False
    )
    raw.to_csv(output_dir / "threshold_mpc_raw.csv", index=False)
    summary.to_csv(output_dir / "threshold_mpc_summary.csv", index=False)
    tests.to_csv(output_dir / "threshold_mpc_network_tests.csv", index=False)
    plot_results(calibration, summary, tests, output_dir)
    write_report(
        calibration,
        summary,
        tests,
        selected_eta,
        selected_lambda,
        split,
        cfg,
        output_dir,
    )
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(cfg),
        "split": split,
        "calibration_end": split // 2,
        "validation_end": split - cfg.horizon,
        "selected_eta": selected_eta,
        "selected_threshold_lambda": selected_lambda,
        "threshold_count": len(thresholds),
        "selection_objective": "2*validation_false_negative_rate + validation_false_discovery_rate",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Selected eta/lambda:", selected_eta, selected_lambda)
    print("\nCalibration\n", calibration.to_string(index=False))
    print("\nHeld-out summary\n", summary.to_string(index=False))
    print("\nNetwork-level tests\n", tests.to_string(index=False))
    print(f"\nResults saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
