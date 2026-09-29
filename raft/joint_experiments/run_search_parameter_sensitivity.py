from __future__ import annotations

import argparse
import json
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
    JointConfig,
    build_cache_replays,
    build_node_replays,
    load_inputs,
    markdown_table,
    run_joint_trials,
    summarize,
)
from run_threshold_aware_mpc_experiment import add_threshold_replay, calibrate_parameters
from run_update_robustness_experiment import holm_adjust


HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent

FACTOR_LABELS = {
    "max_hops": "Maximum forwarding depth",
    "deadline_ms": "Response deadline (ms)",
    "top_k": "Confirmed candidates (top-k)",
}

COLORS = {
    "max_hops": "#0072B2",
    "deadline_ms": "#E69F00",
    "top_k": "#009E73",
}

REFERENCE_LEVELS = {
    "max_hops": 5.0,
    "deadline_ms": 180.0,
    "top_k": 3.0,
}


def scenario_grid() -> list[tuple[str, float]]:
    return (
        [("max_hops", float(x)) for x in (1, 2, 3, 4, 5, 6)]
        + [("deadline_ms", float(x)) for x in (60, 90, 120, 150, 180, 240)]
        + [("top_k", float(x)) for x in (1, 2, 3, 4, 5)]
    )


def run_one_scenario(
    index: int,
    factor: str,
    level: float,
    nodes_actual: np.ndarray,
    replays,
    base_cfg: JointConfig,
) -> tuple[int, str, pd.DataFrame]:
    changes = {
        "max_hops": {"max_hops": int(level)},
        "deadline_ms": {"deadline_ms": float(level)},
        "top_k": {"top_k": int(level)},
    }[factor]
    cfg = replace(base_cfg, **changes)
    raw = run_joint_trials(
        nodes_actual,
        replays,
        cfg,
        pipelines=["threshold_mpc"],
    )
    raw.insert(0, "factor", factor)
    raw.insert(1, "level", level)
    return index, f"{factor}={level:g}", raw


def run_sensitivity(
    nodes_actual: np.ndarray,
    replays,
    cfg: JointConfig,
    workers: int,
) -> pd.DataFrame:
    rows_by_index: dict[int, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                run_one_scenario,
                index,
                factor,
                level,
                nodes_actual,
                replays,
                cfg,
            ): index
            for index, (factor, level) in enumerate(scenario_grid())
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index, label, raw = future.result()
            rows_by_index[index] = raw
            print(f"[{completed:02d}/{len(futures)} completed] {label}", flush=True)
    return pd.concat(
        [rows_by_index[index] for index in range(len(scenario_grid()))],
        ignore_index=True,
    )


def summarize_sensitivity(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (factor, level), frame in raw.groupby(["factor", "level"], sort=False):
        summary = summarize(frame)
        censored = (
            frame.groupby(["workload", "pipeline"], as_index=False)
            .censored_latency_ms.mean()
            .rename(columns={"censored_latency_ms": "mean_censored_latency_ms"})
        )
        summary = summary.merge(censored, on=["workload", "pipeline"], validate="one_to_one")
        summary.insert(0, "factor", factor)
        summary.insert(1, "level", level)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def plotting_frame(raw: pd.DataFrame) -> pd.DataFrame:
    remote = raw[raw.workload == "remote_required"]
    rows = []
    for index, ((factor, level), frame) in enumerate(
        remote.groupby(["factor", "level"], sort=False)
    ):
        rates = frame.groupby("repeat").success.mean().to_numpy() * 100.0
        low, high = bootstrap_mean_ci(rates, seed=510_000 + index)
        rows.append(
            {
                "factor": factor,
                "level": level,
                "success_percent": rates.mean(),
                "ci_low": low,
                "ci_high": high,
                "mean_search_messages": frame.search_messages.mean(),
                "mean_censored_latency_ms": frame.censored_latency_ms.mean(),
                "p95_success_latency_ms": frame.success_latency_ms.quantile(0.95),
                "mean_confirmations": frame.confirmation_attempts.mean(),
                "mean_invalid_candidates": frame.stale_or_invalid_candidates.mean(),
            }
        )
    return pd.DataFrame(rows)


def network_tests(raw: pd.DataFrame) -> pd.DataFrame:
    remote = raw[raw.workload == "remote_required"]
    rows = []
    for factor_idx, (factor, frame) in enumerate(remote.groupby("factor", sort=False)):
        rates = frame.groupby(["level", "repeat"], as_index=False).success.mean()
        reference_level = REFERENCE_LEVELS[factor]
        reference = rates[rates.level == reference_level][["repeat", "success"]].rename(
            columns={"success": "reference_success"}
        )
        for level_idx, level in enumerate(sorted(rates.level.unique())):
            if np.isclose(level, reference_level):
                continue
            alternative = rates[rates.level == level][["repeat", "success"]].rename(
                columns={"success": "alternative_success"}
            )
            pair = reference.merge(alternative, on="repeat", validate="one_to_one")
            difference = 100.0 * (
                pair.alternative_success - pair.reference_success
            ).to_numpy()
            low, high = bootstrap_mean_ci(
                difference,
                seed=520_000 + factor_idx * 100 + level_idx,
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
                    "reference_level": reference_level,
                    "alternative_level": level,
                    "network_clusters": len(pair),
                    "reference_success_percent": 100.0 * pair.reference_success.mean(),
                    "alternative_success_percent": 100.0 * pair.alternative_success.mean(),
                    "alternative_minus_reference_pp": difference.mean(),
                    "cluster_bootstrap_ci_low_pp": low,
                    "cluster_bootstrap_ci_high_pp": high,
                    "wilcoxon_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = holm_adjust(result.wilcoxon_p.to_numpy())
    return result


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
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), constrained_layout=True)
    specifications = [
        ("max_hops", "Maximum forwarding depth, Dmax", "(a) Search-depth sensitivity"),
        ("deadline_ms", "Response deadline (ms)", "(b) Deadline sensitivity"),
        ("top_k", "Confirmed candidates (top-k)", "(c) Confirmation sensitivity"),
    ]
    for ax, (factor, xlabel, title) in zip(axes.ravel()[:3], specifications):
        data = plot_data[plot_data.factor == factor].sort_values("level")
        x = data.level.to_numpy()
        ax.plot(
            x,
            data.success_percent,
            color=COLORS[factor],
            marker="o",
            linewidth=1.4,
            markersize=4,
        )
        ax.fill_between(
            x,
            data.ci_low,
            data.ci_high,
            color=COLORS[factor],
            alpha=0.15,
            linewidth=0,
        )
        reference = REFERENCE_LEVELS[factor]
        ax.axvline(reference, color="#555555", linestyle="--", linewidth=0.8)
        ax.text(
            reference,
            4,
            "preset",
            rotation=90,
            va="bottom",
            ha="right",
            color="#555555",
            fontsize=6.5,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Remote candidate discovery (%)")
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    markers = {"max_hops": "o", "deadline_ms": "s", "top_k": "^"}
    start_labels = {
        "max_hops": "D=1",
        "deadline_ms": "60 ms",
        "top_k": "k=1",
    }
    for factor in FACTOR_LABELS:
        data = plot_data[plot_data.factor == factor].sort_values("level")
        ax.plot(
            data.mean_search_messages,
            data.success_percent,
            color=COLORS[factor],
            marker=markers[factor],
            markersize=5,
            linewidth=1.0,
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=FACTOR_LABELS[factor],
        )
        row = data.iloc[0]
        ax.annotate(
            start_labels[factor],
            (row.mean_search_messages, row.success_percent),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
        )
    preset_rows = []
    for factor, reference in REFERENCE_LEVELS.items():
        preset_rows.append(
            plot_data[(plot_data.factor == factor) & np.isclose(plot_data.level, reference)].iloc[0]
        )
    preset_x = float(np.mean([row.mean_search_messages for row in preset_rows]))
    preset_y = float(np.mean([row.success_percent for row in preset_rows]))
    ax.scatter(preset_x, preset_y, marker="*", s=65, color="black", zorder=5)
    ax.annotate("preset", (preset_x, preset_y), xytext=(-28, 6), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Mean search messages per task")
    ax.set_ylabel("Remote candidate discovery (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(d) Discovery–communication trade-off", loc="left", fontweight="bold")
    ax.grid(color="#D9D9D9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")
    export_figure(fig, output_dir, "search_parameter_sensitivity")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    cfg: JointConfig,
    selected_eta: float,
    selected_lambda: float,
    output_dir: Path,
) -> None:
    remote = summary[summary.workload == "remote_required"].copy()
    table_columns = [
        "factor",
        "level",
        "success_percent",
        "success_ci95_percent",
        "mean_censored_latency_ms",
        "p95_success_latency_ms",
        "mean_search_messages",
        "mean_confirmations",
        "mean_invalid_candidates",
        "cache_blocked_discovery_percent",
        "protocol_failure_with_candidate_percent",
    ]
    remote_indexed = remote.set_index(["factor", "level"])
    d4 = remote_indexed.loc[("max_hops", 4.0)]
    d5 = remote_indexed.loc[("max_hops", 5.0)]
    d6 = remote_indexed.loc[("max_hops", 6.0)]
    deadline_180 = remote_indexed.loc[("deadline_ms", 180.0)]
    deadline_240 = remote_indexed.loc[("deadline_ms", 240.0)]
    top1 = remote_indexed.loc[("top_k", 1.0)]
    top2 = remote_indexed.loc[("top_k", 2.0)]
    top3 = remote_indexed.loc[("top_k", 3.0)]
    d4_test = tests[(tests.factor == "max_hops") & np.isclose(tests.alternative_level, 4.0)].iloc[0]
    top2_test = tests[(tests.factor == "top_k") & np.isclose(tests.alternative_level, 2.0)].iloc[0]
    deadline240_test = tests[(tests.factor == "deadline_ms") & np.isclose(tests.alternative_level, 240.0)].iloc[0]
    report = f"""# 搜索协议参数敏感性实验

## 1. 实验设计

- 最大转发深度Dmax：1、2、3、4、5和6，预设值为5。
- 最大响应时间：60、90、120、150、180和240 ms，预设值为180 ms。
- 二次确认数量top-k：1、2、3、4和5，预设值为3。
- 单因素扫描时，其余搜索参数固定为Dmax={cfg.max_hops}、响应时间={cfg.deadline_ms:.0f} ms、top-k={cfg.top_k}。
- 更新层固定为阈值感知MPC-DP，更新预算为{cfg.update_budget:.0%}，eta={selected_eta:g}、lambda={selected_lambda:g}；没有根据搜索测试结果重新训练预测器或更新调度器。
- 每个条件包含{cfg.network_repeats}个独立网络，每个网络每类工作负载{cfg.tasks_per_network}个配对任务；网络、任务、资源需求和动态故障种子在参数条件间保持一致。

## 2. 主要发现与设置指导

- `Dmax=1`无法越过直接业务邻居，因此远端任务发现率为0；`Dmax=3、4、5`分别达到79.40%、{d4.success_percent:.2f}%和{d5.success_percent:.2f}%。`Dmax=4`比预设值5低{d5.success_percent - d4.success_percent:.2f}个百分点，Holm校正后P={d4_test.holm_adjusted_p:.3f}；`Dmax=6`仍为{d6.success_percent:.2f}%，没有额外收益。因此5是偏可靠性的保守设置，4是略省消息的可选设置。
- 响应时间从180 ms延长至240 ms使发现率由{deadline_180.success_percent:.2f}%升至{deadline_240.success_percent:.2f}%（+{deadline_240.success_percent - deadline_180.success_percent:.2f}个百分点，Holm校正P={deadline240_test.holm_adjusted_p:.4f}），但平均截尾时延增加{deadline_240.mean_censored_latency_ms - deadline_180.mean_censored_latency_ms:.2f} ms，成功搜索P95时延增加{deadline_240.p95_success_latency_ms - deadline_180.p95_success_latency_ms:.2f} ms。因此180 ms是时延—成功率折中点，240 ms适用于更重视发现率的宽松时延场景。
- `top-k=1`的发现率仅{top1.success_percent:.2f}%，比top-3低{top3.success_percent - top1.success_percent:.2f}个百分点；top-2达到{top2.success_percent:.2f}%，与top-3仅差{top3.success_percent - top2.success_percent:.2f}个百分点且Holm校正P={top2_test.holm_adjusted_p:.3f}。top-4和top-5均未超过top-3。因此top-3是保守确认设置，资源更紧张时可考虑top-2。
- 参数指导不是“越大越好”：更深转发、更多确认和更长截止时间在饱和后主要增加搜索消息或等待时间。

## 3. 远端候选发现结果

{markdown_table(remote, table_columns)}

## 4. 相对预设值的网络级配对检验

{markdown_table(tests, ['factor', 'reference_level', 'alternative_level', 'network_clusters', 'reference_success_percent', 'alternative_success_percent', 'alternative_minus_reference_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_p', 'holm_adjusted_p'])}

## 5. 使用边界

1. 本实验用于解释预设参数的敏感性，不使用测试结果反向修改任务或选择性删除失败条件。
2. Holm校正覆盖全部相对预设值的网络级比较；不显著不能解释为已证明等效。
3. 平均截尾时延将搜索失败按该条件截止时间计入，成功时延P95仅统计成功搜索，两者含义不同。
4. 结果仍来自轨迹驱动的动态网络仿真，实际设备与网络部署需要单独验证。
"""
    (output_dir / "搜索协议参数敏感性实验结果与分析.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "search_parameter_sensitivity_formal",
    )
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--tasks-per-network", type=int, default=80)
    parser.add_argument("--workers", type=int, default=6)
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
    raw = run_sensitivity(nodes_actual, replays, cfg, args.workers)
    summary = summarize_sensitivity(raw)
    tests = network_tests(raw)
    plot_data = plotting_frame(raw)
    raw.to_csv(args.output_dir / "search_parameter_raw.csv", index=False)
    summary.to_csv(args.output_dir / "search_parameter_summary.csv", index=False)
    tests.to_csv(args.output_dir / "search_parameter_network_tests.csv", index=False)
    plot_data.to_csv(args.output_dir / "search_parameter_plot_data.csv", index=False)
    plot_results(plot_data, args.output_dir)
    write_report(summary, tests, cfg, selected_eta, selected_lambda, args.output_dir)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(cfg),
        "scenarios": [{"factor": factor, "level": level} for factor, level in scenario_grid()],
        "reference_levels": REFERENCE_LEVELS,
        "selected_eta": selected_eta,
        "selected_threshold_lambda": selected_lambda,
        "test_policy": "one-factor-at-a-time paired network/task sensitivity analysis",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = summary[summary.workload == "remote_required"][[
        "factor",
        "level",
        "success_percent",
        "mean_censored_latency_ms",
        "mean_search_messages",
    ]]
    print("\nRemote search-parameter summary\n", concise.to_string(index=False))
    print(f"\nResults saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
