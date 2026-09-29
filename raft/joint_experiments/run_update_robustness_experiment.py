from __future__ import annotations

import argparse
import json
import math
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_cluster_and_gossip_budget_analysis import bootstrap_mean_ci
from run_joint_discovery_experiment import (
    BudgetGuard,
    JointConfig,
    build_cache_replays,
    build_node_replays,
    load_inputs,
    markdown_table,
    run_joint_trials,
    summarize,
    weighted_squared_error,
)
from run_threshold_aware_mpc_experiment import (
    add_threshold_replay,
    calibrate_parameters,
)


HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent

PIPELINES = [
    "threshold_mpc",
    "proposed",
    "last_mpc",
    "fits_periodic",
    "fits_delta",
]

LABELS = {
    "threshold_mpc": "Threshold-aware MPC-DP",
    "proposed": "Original MPC-DP",
    "last_mpc": "Last-value + MPC-DP",
    "fits_periodic": "Periodic",
    "fits_delta": "Send-on-delta",
}

COLORS = {
    "threshold_mpc": "#0072B2",
    "proposed": "#56B4E9",
    "last_mpc": "#CC79A7",
    "fits_periodic": "#E69F00",
    "fits_delta": "#D55E00",
}


def replay_delta_for_calibration(
    actual: np.ndarray,
    initial_cache: np.ndarray,
    budget: float,
    threshold: float,
) -> np.ndarray:
    cache = np.asarray(initial_cache, dtype=float).copy()
    history = np.empty_like(actual)
    guard = BudgetGuard(len(actual), budget)
    for t in range(len(actual)):
        delta = math.sqrt(weighted_squared_error(actual[t], cache))
        action = guard.apply(delta > threshold, t)
        if action:
            cache = actual[t].copy()
        history[t] = cache
    if guard.used != guard.target:
        raise AssertionError(f"delta calibration used {guard.used}, expected {guard.target}")
    return history


def calibrate_delta_threshold(actual: np.ndarray, split: int, budget: float, horizon: int) -> float:
    calibration_end = split // 2
    validation_end = split - horizon
    calibration = actual[:calibration_end]
    validation = actual[calibration_end:validation_end]
    one_step = np.sqrt(
        np.sum(0.5 * np.square(np.diff(calibration, axis=0)), axis=1)
    )
    candidates = np.unique(
        np.concatenate(
            ([0.0], np.quantile(one_step, np.linspace(0.05, 0.95, 19)), [np.inf])
        )
    )
    rows = []
    for threshold in candidates:
        cache = replay_delta_for_calibration(
            validation,
            actual[calibration_end - 1],
            budget,
            float(threshold),
        )
        wrmse = float(
            np.sqrt(np.mean(np.sum(0.5 * np.square(validation - cache), axis=1)))
        )
        rows.append((wrmse, float(threshold)))
    return min(rows, key=lambda item: (item[0], item[1]))[1]


def scenario_grid() -> list[tuple[str, float, float, float]]:
    budgets = (0.25, 0.375, 0.50, 0.625, 0.75)
    disruptions = (0.0, 0.05, 0.10, 0.20, 0.30)
    scenarios = [("update_budget", value, value, 0.05, 0.05) for value in budgets]
    scenarios += [("outage_rate", value, 0.50, value, 0.05) for value in disruptions]
    scenarios += [("topology_churn_rate", value, 0.50, 0.05, value) for value in disruptions]
    return scenarios


def run_sensitivity(
    actual: np.ndarray,
    predictions: np.ndarray,
    base_cfg: JointConfig,
    selected_eta: float,
    selected_lambda: float,
    thresholds: np.ndarray,
    workers: int,
) -> pd.DataFrame:
    delta_by_budget: dict[float, float] = {}
    for _, _, budget, _, _ in scenario_grid():
        if budget not in delta_by_budget:
            delta_by_budget[budget] = calibrate_delta_threshold(
                actual, base_cfg.trace_start, budget, base_cfg.horizon
            )
    jobs = [
        (
            scenario,
            actual,
            predictions,
            base_cfg,
            selected_eta,
            selected_lambda,
            thresholds,
            delta_by_budget[scenario[2]],
        )
        for scenario in scenario_grid()
    ]
    rows_by_index: dict[int, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(run_one_scenario, index, *job): index
            for index, job in enumerate(jobs)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index, label, raw = future.result()
            rows_by_index[index] = raw
            print(f"[{completed:02d}/{len(jobs)} completed] {label}", flush=True)
    rows = [rows_by_index[index] for index in range(len(jobs))]
    return pd.concat(rows, ignore_index=True)


def run_one_scenario(
    index: int,
    scenario: tuple[str, float, float, float, float],
    actual: np.ndarray,
    predictions: np.ndarray,
    base_cfg: JointConfig,
    selected_eta: float,
    selected_lambda: float,
    thresholds: np.ndarray,
    delta_threshold: float,
) -> tuple[int, str, pd.DataFrame]:
    factor, level, budget, outage, churn = scenario
    cfg = replace(
        base_cfg,
        update_budget=budget,
        outage_rate=outage,
        topology_churn_rate=churn,
        delta_threshold=delta_threshold,
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
    raw = run_joint_trials(nodes_actual, replays, cfg, pipelines=PIPELINES)
    raw.insert(0, "factor", factor)
    raw.insert(1, "level", level)
    raw.insert(2, "configured_budget", budget)
    raw.insert(3, "configured_outage", outage)
    raw.insert(4, "configured_churn", churn)
    raw.insert(5, "delta_threshold", delta_threshold)
    label = (
        f"{factor}={level:g}, budget={budget:g}, outage={outage:g}, churn={churn:g}"
    )
    return index, label, raw


def summarize_sensitivity(raw: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (factor, level), frame in raw.groupby(["factor", "level"], sort=False):
        result = summarize(frame)
        result.insert(0, "factor", factor)
        result.insert(1, "level", level)
        summaries.append(result)
    return pd.concat(summaries, ignore_index=True)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def network_tests(raw: pd.DataFrame) -> pd.DataFrame:
    comparators = [p for p in PIPELINES if p != "threshold_mpc"]
    rows = []
    remote = raw[raw.workload == "remote_required"]
    for factor_idx, ((factor, level), frame) in enumerate(
        remote.groupby(["factor", "level"], sort=False)
    ):
        rates = frame.groupby(["repeat", "pipeline"], as_index=False).success.mean()
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
                seed=310_000 + factor_idx * 100 + comparator_idx,
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
                    "factor": factor,
                    "level": level,
                    "comparison": f"threshold_mpc vs {comparator}",
                    "network_clusters": len(pair),
                    "threshold_mpc_percent": 100.0 * pair.threshold_mpc.mean(),
                    "baseline_percent": 100.0 * pair.baseline.mean(),
                    "paired_difference_pp": difference.mean(),
                    "cluster_bootstrap_ci_low_pp": ci_low,
                    "cluster_bootstrap_ci_high_pp": ci_high,
                    "wilcoxon_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = holm_adjust(result.wilcoxon_p.to_numpy())
    return result


def plotting_frame(raw: pd.DataFrame) -> pd.DataFrame:
    remote = raw[raw.workload == "remote_required"]
    rows = []
    grouped = remote.groupby(["factor", "level", "pipeline"], sort=False)
    for index, ((factor, level, pipeline), frame) in enumerate(grouped):
        network_rates = frame.groupby("repeat").success.mean().to_numpy() * 100.0
        low, high = bootstrap_mean_ci(network_rates, seed=410_000 + index)
        rows.append(
            {
                "factor": factor,
                "level": level,
                "pipeline": pipeline,
                "success_percent": network_rates.mean(),
                "ci_low": low,
                "ci_high": high,
                "cache_wrmse": frame.cache_wrmse.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def export_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def plot_results(plot_data: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    factor_specs = [
        ("update_budget", "Update budget (%)", "(a) Communication-budget sensitivity"),
        ("outage_rate", "Node outage rate (%)", "(b) Node-failure robustness"),
        ("topology_churn_rate", "Changed-link probability (%)", "(c) Topology-change robustness"),
    ]
    for ax, (factor, xlabel, title) in zip(axes.ravel()[:3], factor_specs):
        subset = plot_data[plot_data.factor == factor]
        for pipeline in PIPELINES:
            line = subset[subset.pipeline == pipeline].sort_values("level")
            x = 100.0 * line.level.to_numpy()
            y = line.success_percent.to_numpy()
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color=COLORS[pipeline],
                label=LABELS[pipeline],
            )
            ax.fill_between(
                x,
                line.ci_low.to_numpy(),
                line.ci_high.to_numpy(),
                color=COLORS[pipeline],
                alpha=0.10,
                linewidth=0,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Remote candidate discovery (%)")
        ax.set_ylim(50, 90)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    budget = plot_data[plot_data.factor == "update_budget"]
    ax = axes[1, 1]
    for pipeline in PIPELINES:
        line = budget[budget.pipeline == pipeline].sort_values("level")
        ax.plot(
            100.0 * line.level,
            line.cache_wrmse,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color=COLORS[pipeline],
            label=LABELS[pipeline],
        )
    ax.set_xlabel("Update budget (%)")
    ax.set_ylabel("State-cache WRMSE")
    ax.set_title("(d) Cache fidelity under equal budgets", loc="left", fontweight="bold")
    ax.grid(color="#D9D9D9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=5,
        frameon=False,
    )
    export_figure(fig, output_dir, "update_robustness")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    selected_eta: float,
    selected_lambda: float,
    cfg: JointConfig,
    output_dir: Path,
) -> None:
    remote = summary[summary.workload == "remote_required"].copy()
    remote["pipeline"] = remote.pipeline.map(LABELS)
    table_columns = [
        "factor",
        "level",
        "pipeline",
        "success_percent",
        "candidate_micro_recall",
        "candidate_micro_precision",
        "cache_blocked_discovery_percent",
        "protocol_failure_with_candidate_percent",
        "cache_wrmse",
        "mean_search_messages",
        "mean_update_messages",
    ]
    vs_original = tests[tests.comparison == "threshold_mpc vs proposed"]
    vs_periodic = tests[tests.comparison == "threshold_mpc vs fits_periodic"]
    budget_threshold = remote[
        (remote.factor == "update_budget")
        & (remote.pipeline == LABELS["threshold_mpc"])
    ].sort_values("level")
    outage_threshold = remote[
        (remote.factor == "outage_rate")
        & (remote.pipeline == LABELS["threshold_mpc"])
    ].sort_values("level")
    churn_threshold = remote[
        (remote.factor == "topology_churn_rate")
        & (remote.pipeline == LABELS["threshold_mpc"])
    ].sort_values("level")
    best_budget_count = 0
    original_remote = summary[summary.workload == "remote_required"]
    for level, frame in original_remote[original_remote.factor == "update_budget"].groupby("level"):
        threshold_success = float(frame[frame.pipeline == "threshold_mpc"].success_percent.iloc[0])
        if np.isclose(threshold_success, frame.success_percent.max()):
            best_budget_count += 1
    report = f"""# 更新策略动态鲁棒性实验

## 1. 预先固定的实验设计

- 更新预算：25%、37.5%、50%、62.5%和75%；此扫描固定节点断联率与拓扑变化率为5%。
- 节点断联率：0%、5%、10%、20%和30%；此扫描固定更新预算为50%、拓扑变化率为5%。
- 拓扑变化率：0%、5%、10%、20%和30%；此扫描固定更新预算为50%、节点断联率为5%。
- 每个条件包含{cfg.network_repeats}个独立网络，每个网络每类工作负载{cfg.tasks_per_network}个配对任务；同一条件内所有方法共享网络、任务和故障随机种子。
- 阈值感知MPC-DP冻结先前验证段选出的eta={selected_eta:g}、lambda={selected_lambda:g}，没有按测试条件重新调参。
- Send-on-delta仅使用测试前验证段，按每种更新预算选择使验证WRMSE最低的触发阈值。
- 所有可部署方法在每个预算条件下使用完全相同的实际更新比例。

## 2. 主要发现

- 阈值感知MPC-DP在15/15个条件下均高于原MPC-DP，平均差值范围为{vs_original.paired_difference_pp.min():.2f}—{vs_original.paired_difference_pp.max():.2f}个百分点；Holm校正后仍有{int((vs_original.holm_adjusted_p < 0.05).sum())}/15个条件显著。
- 相比周期更新，差值范围为{vs_periodic.paired_difference_pp.min():.2f}—{vs_periodic.paired_difference_pp.max():.2f}个百分点；Holm校正后0/15个条件显著，即本轮灵敏度实验未检测到显著差异，但这不能被解释为已证明两者等效，也不支持“全面优于周期更新”。
- 在五档更新预算中，阈值感知MPC-DP有{best_budget_count}/5档取得最高发现率；其发现率由{budget_threshold.success_percent.iloc[0]:.2f}%升至{budget_threshold.success_percent.iloc[-1]:.2f}%，缓存WRMSE由{budget_threshold.cache_wrmse.iloc[0]:.3f}降至{budget_threshold.cache_wrmse.iloc[-1]:.3f}。
- 节点断联率从0%升至30%时，阈值感知方法的发现率下降{outage_threshold.success_percent.iloc[0] - outage_threshold.success_percent.iloc[-1]:.2f}个百分点；拓扑变化率从0%升至30%时下降{churn_threshold.success_percent.iloc[0] - churn_threshold.success_percent.iloc[-1]:.2f}个百分点。当前系统对业务拓扑变化比对节点中途断联更敏感。

## 3. 远端候选发现结果

{markdown_table(remote, table_columns)}

## 4. 阈值感知MPC-DP的网络级配对检验

{markdown_table(tests, ['factor', 'level', 'comparison', 'network_clusters', 'threshold_mpc_percent', 'baseline_percent', 'paired_difference_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_p', 'holm_adjusted_p'])}

## 5. 解释边界

1. 灵敏度检验属于预设的单因素扫描，不等同于所有参数的完整笛卡尔积。
2. Holm校正覆盖本表全部网络级比较；未校正和校正后的P值均保留。
3. 动态故障由仿真注入，不能替代真实工业边缘设备和网络部署。
4. 若周期更新在发现率上保持优势，应据实表述为缓存误差—候选发现之间的性能权衡，不能选择性删除该基线。
5. 前一轮40网络、每网络120任务的50%预算单点实验中，周期更新比阈值感知方法高1.42个百分点且未校正P=0.0198；本轮较小网络级样本在Holm校正后不显著。两轮结果方向一致，应共同报告，不能用本轮“不显著”覆盖前一轮结果。
"""
    (output_dir / "更新策略动态鲁棒性实验结果与分析.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "update_robustness_formal",
    )
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--tasks-per-network", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = JointConfig(
        network_repeats=args.repeats,
        tasks_per_network=args.tasks_per_network,
    )
    actual, predictions = load_inputs(args.data_dir)
    selected_eta, selected_lambda, thresholds, _ = calibrate_parameters(
        actual,
        predictions,
        cfg.trace_start,
        cfg.update_budget,
        cfg.horizon,
    )
    raw = run_sensitivity(
        actual,
        predictions,
        cfg,
        selected_eta,
        selected_lambda,
        thresholds,
        args.workers,
    )
    summary = summarize_sensitivity(raw)
    tests = network_tests(raw)
    plot_data = plotting_frame(raw)
    raw.to_csv(args.output_dir / "update_robustness_raw.csv", index=False)
    summary.to_csv(args.output_dir / "update_robustness_summary.csv", index=False)
    tests.to_csv(args.output_dir / "update_robustness_network_tests.csv", index=False)
    plot_data.to_csv(args.output_dir / "update_robustness_plot_data.csv", index=False)
    plot_results(plot_data, args.output_dir)
    write_report(summary, tests, selected_eta, selected_lambda, cfg, args.output_dir)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(cfg),
        "scenarios": [
            {
                "factor": factor,
                "level": level,
                "update_budget": budget,
                "outage_rate": outage,
                "topology_churn_rate": churn,
            }
            for factor, level, budget, outage, churn in scenario_grid()
        ],
        "selected_eta": selected_eta,
        "selected_threshold_lambda": selected_lambda,
        "test_policy": "one-factor-at-a-time, paired tasks and stochastic environments",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = summary[
        (summary.workload == "remote_required")
        & (summary.pipeline.isin(["threshold_mpc", "proposed", "fits_periodic"]))
    ][["factor", "level", "pipeline", "success_percent", "cache_wrmse"]]
    print("\nRemote robustness summary\n", concise.to_string(index=False))
    print(f"\nResults saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
