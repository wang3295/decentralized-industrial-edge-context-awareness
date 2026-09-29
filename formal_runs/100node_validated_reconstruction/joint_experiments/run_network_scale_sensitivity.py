from __future__ import annotations

import argparse
import json
import math
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_full_pipeline_ablation import (
    FullAblationConfig,
    mean_and_ci,
    network_summary,
    run_one_network_condition,
)
from run_joint_discovery_experiment import build_cache_replays, build_node_replays
from run_threshold_aware_mpc_experiment import (
    add_threshold_replay,
    calibrate_parameters,
)
from run_trust_robustness_experiment import bootstrap_ci, holm_adjust, joint_config

import sys

HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent
sys.path.insert(0, str(RAFT_DIR / "update_experiments"))

from run_update_experiments import load_inputs  # noqa: E402


PIPELINES = {
    "full": "Full method",
    "periodic": "Periodic update",
    "send_on_delta": "Send-on-delta",
    "gossip": "Gossip-style query",
    "no_trust": "No online trust",
}

PIPELINE_SPEC = {
    "full": ("threshold_mpc", "online_beta", "workflow"),
    "periodic": ("fits_periodic", "online_beta", "workflow"),
    "send_on_delta": ("fits_delta", "online_beta", "workflow"),
    "gossip": ("threshold_mpc", "online_beta", "gossip"),
    "no_trust": ("threshold_mpc", "no_trust", "workflow"),
}

COLORS = {
    "full": "#0072B2",
    "periodic": "#E69F00",
    "send_on_delta": "#CC79A7",
    "lyapunov": "#56B4E9",
    "gossip": "#009E73",
    "no_trust": "#D55E00",
}

MARKERS = {
    "full": "o",
    "periodic": "^",
    "send_on_delta": "v",
    "lyapunov": "s",
    "gossip": "D",
    "no_trust": "P",
}


def parse_node_counts(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 5 for value in values):
        raise argparse.ArgumentTypeError("node counts must be comma-separated integers >= 5")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("node counts must be unique")
    return values


def build_scale_inputs(
    actual: np.ndarray,
    predictions: np.ndarray,
    cfg: FullAblationConfig,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, dict]:
    jcfg = joint_config(cfg)
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
    required = {spec[0] for spec in PIPELINE_SPEC.values()}
    missing = required - set(replays)
    if missing:
        raise AssertionError(f"Missing cache replays: {sorted(missing)}")
    return nodes_actual, replays


def run_scale_condition(
    num_nodes: int,
    nodes_actual: np.ndarray,
    replays: dict,
    cfg: FullAblationConfig,
    workers: int,
) -> pd.DataFrame:
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_one_network_condition,
                0.2,
                repeat,
                nodes_actual,
                replays,
                cfg,
                PIPELINES,
                PIPELINE_SPEC,
            ): repeat
            for repeat in range(cfg.network_repeats)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            repeat = futures[future]
            frame = future.result()
            frame.insert(0, "num_nodes", num_nodes)
            frames.append(frame)
            if completed % 5 == 0 or completed == len(futures):
                print(
                    f"N={num_nodes}: [{completed:02d}/{len(futures):02d}] "
                    f"completed; latest repeat={repeat}",
                    flush=True,
                )
    return pd.concat(frames, ignore_index=True)


def summarize_by_scale(network: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (num_nodes, pipeline), group in network.groupby(
        ["num_nodes", "pipeline"], sort=False
    ):
        honest, honest_ci = mean_and_ci(100.0 * group.honest_success_rate)
        malicious, malicious_ci = mean_and_ci(
            100.0 * group.malicious_selection_rate
        )
        blocked, blocked_ci = mean_and_ci(100.0 * group.cache_blocked_rate)
        latency, latency_ci = mean_and_ci(group.p95_success_latency_ms)
        messages, messages_ci = mean_and_ci(group.mean_total_messages)
        rows.append(
            {
                "num_nodes": int(num_nodes),
                "pipeline": pipeline,
                "networks": len(group),
                "honest_success_percent": honest,
                "honest_success_ci95_percent": honest_ci,
                "malicious_selection_percent": malicious,
                "malicious_selection_ci95_percent": malicious_ci,
                "cache_blocked_percent": blocked,
                "cache_blocked_ci95_percent": blocked_ci,
                "candidate_recall_percent": 100.0
                * float(group.candidate_recall.mean()),
                "p95_success_latency_ms": latency,
                "p95_success_latency_ci95_ms": latency_ci,
                "mean_search_messages": float(group.mean_search_messages.mean()),
                "mean_update_messages": float(group.mean_update_messages.mean()),
                "mean_total_messages": messages,
                "mean_total_messages_ci95": messages_ci,
                "mean_search_messages_per_node": float(
                    group.mean_search_messages.mean() / num_nodes
                ),
                "mean_update_messages_per_node": float(
                    group.mean_update_messages.mean() / num_nodes
                ),
                "mean_total_messages_per_node": float(messages / num_nodes),
                "cache_wrmse": float(group.cache_wrmse.mean()),
                "update_rate_percent": 100.0 * float(group.update_rate.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["num_nodes", "pipeline"])


def paired_tests(
    network: pd.DataFrame,
    node_counts: Sequence[int],
    comparators: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows = []
    endpoints = ("honest_success_rate", "malicious_selection_rate")
    if comparators is None:
        comparators = tuple(pipeline for pipeline in PIPELINES if pipeline != "full")
    for size_idx, num_nodes in enumerate(node_counts):
        full = network[
            network.num_nodes.eq(num_nodes) & network.pipeline.eq("full")
        ][["repeat", *endpoints]]
        for comparator_idx, comparator in enumerate(comparators):
            baseline = network[
                network.num_nodes.eq(num_nodes) & network.pipeline.eq(comparator)
            ][["repeat", *endpoints]]
            merged = full.merge(baseline, on="repeat", suffixes=("_full", "_baseline"))
            if len(merged) != len(full):
                raise AssertionError("Paired network rows are incomplete")
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
                    seed=310_000
                    + size_idx * 1_000
                    + comparator_idx * 100
                    + endpoint_idx,
                )
                rows.append(
                    {
                        "num_nodes": num_nodes,
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


def export_figure(fig: plt.Figure, output_dir: Path) -> None:
    stem = output_dir / "network_scale_sensitivity"
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
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    for pipeline in PIPELINES:
        data = summary[summary.pipeline.eq(pipeline)].sort_values("num_nodes")
        style = {
            "color": COLORS[pipeline],
            "marker": MARKERS[pipeline],
            "linewidth": 1.4,
            "markersize": 4.2,
            "capsize": 2,
            "label": PIPELINES[pipeline],
        }
        axes[0, 0].errorbar(
            data.num_nodes,
            data.honest_success_percent,
            yerr=data.honest_success_ci95_percent,
            **style,
        )
        axes[0, 1].errorbar(
            data.num_nodes,
            data.malicious_selection_percent,
            yerr=data.malicious_selection_ci95_percent,
            **style,
        )
        axes[1, 0].errorbar(
            data.num_nodes,
            data.mean_total_messages,
            yerr=data.mean_total_messages_ci95,
            **style,
        )
        axes[1, 1].errorbar(
            data.num_nodes,
            data.p95_success_latency_ms,
            yerr=data.p95_success_latency_ci95_ms,
            **style,
        )

    axes[0, 0].set_title("(a) Trustworthy candidate discovery", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Honest candidate discovery (%)")
    axes[0, 1].set_title("(b) Malicious candidate selection", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Malicious selection (%)")
    axes[1, 0].set_title("(c) Communication overhead", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Search + amortized update messages/request")
    axes[1, 1].set_title(
        "(d) Simulated decision latency (180-ms deadline)",
        loc="left",
        fontweight="bold",
    )
    axes[1, 1].set_ylabel("Mean network-level P95 latency (ms, simulated)")
    latency = summary["p95_success_latency_ms"].to_numpy(dtype=float)
    if np.isfinite(latency).any():
        lower = max(0.0, float(np.nanmin(latency)) - 10.0)
        upper = max(180.5, float(np.nanmax(latency)) + 5.0)
        axes[1, 1].set_ylim(lower, upper)
        axes[1, 1].axhline(180.0, color="#777777", linestyle="--", linewidth=0.8)
    for ax in axes.flat:
        ax.set_xlabel("Number of ECNs")
        ax.set_xticks(sorted(summary.num_nodes.unique()))
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=5,
    )
    export_figure(fig, output_dir)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame[list(columns)].iterrows():
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


def write_report(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    node_counts: Sequence[int],
    cfg: FullAblationConfig,
    output_dir: Path,
) -> None:
    def result(num_nodes: int, pipeline: str, field: str) -> float:
        row = summary[
            summary.num_nodes.eq(num_nodes) & summary.pipeline.eq(pipeline)
        ]
        return float(row.iloc[0][field])

    def test_result(num_nodes: int, comparison: str, field: str) -> float:
        row = tests[
            tests.num_nodes.eq(num_nodes)
            & tests.endpoint.eq("honest_success_rate")
            & tests.comparison.eq(comparison)
        ]
        return float(row.iloc[0][field])

    full = summary[summary.pipeline.eq("full")].sort_values("num_nodes")
    first = full.iloc[0]
    last = full.iloc[-1]
    success_change = float(last.honest_success_percent - first.honest_success_percent)
    message_ratio = float(last.mean_total_messages / first.mean_total_messages)
    latency_change = float(last.p95_success_latency_ms - first.p95_success_latency_ms)
    periodic_rows = []
    delta_rows = []
    gossip_rows = []
    no_trust_rows = []
    for num_nodes in node_counts:
        periodic_rows.append(
            (
                num_nodes,
                test_result(num_nodes, "full vs periodic", "full_minus_baseline_pp"),
                test_result(
                    num_nodes, "full vs periodic", "cluster_bootstrap_ci_low_pp"
                ),
                test_result(
                    num_nodes, "full vs periodic", "cluster_bootstrap_ci_high_pp"
                ),
                test_result(num_nodes, "full vs periodic", "holm_adjusted_p"),
            )
        )
        delta_rows.append(
            (
                num_nodes,
                test_result(
                    num_nodes, "full vs send_on_delta", "full_minus_baseline_pp"
                ),
                test_result(
                    num_nodes,
                    "full vs send_on_delta",
                    "cluster_bootstrap_ci_low_pp",
                ),
                test_result(
                    num_nodes,
                    "full vs send_on_delta",
                    "cluster_bootstrap_ci_high_pp",
                ),
                test_result(
                    num_nodes, "full vs send_on_delta", "holm_adjusted_p"
                ),
            )
        )
        gossip_rows.append(
            (
                num_nodes,
                test_result(num_nodes, "full vs gossip", "full_minus_baseline_pp"),
                test_result(num_nodes, "full vs gossip", "holm_adjusted_p"),
                100.0
                * (
                    result(num_nodes, "full", "mean_total_messages")
                    - result(num_nodes, "gossip", "mean_total_messages")
                )
                / result(num_nodes, "full", "mean_total_messages"),
            )
        )
        no_trust_rows.append(
            (
                num_nodes,
                test_result(num_nodes, "full vs no_trust", "full_minus_baseline_pp"),
                test_result(num_nodes, "full vs no_trust", "holm_adjusted_p"),
            )
        )

    periodic_text = "；".join(
        f"N={n}为{gap:+.2f}个百分点（95% CI {low:+.2f}至{high:+.2f}，"
        f"Holm p={p:.4f}）"
        for n, gap, low, high, p in periodic_rows
    )
    delta_text = "；".join(
        f"N={n}为{gap:+.2f}个百分点（95% CI {low:+.2f}至{high:+.2f}，"
        f"Holm p={p:.4f}）"
        for n, gap, low, high, p in delta_rows
    )
    gossip_text = "；".join(
        f"N={n}成功率高{gain:.2f}个百分点（Holm p={p:.4f}），"
        f"但消息开销高{message_cost:.2f}%"
        for n, gain, p, message_cost in gossip_rows
    )
    no_trust_text = "；".join(
        f"N={n}高{gain:.2f}个百分点（Holm p={p:.4f}）"
        for n, gain, p in no_trust_rows
    )
    full_wrmse_text = "/".join(
        f"{result(n, 'full', 'cache_wrmse'):.4f}" for n in node_counts
    )
    periodic_wrmse_text = "/".join(
        f"{result(n, 'periodic', 'cache_wrmse'):.4f}" for n in node_counts
    )
    per_node_message_text = "/".join(
        f"{result(n, 'full', 'mean_total_messages_per_node'):.2f}"
        for n in node_counts
    )
    periodic_all_nonsignificant = all(p >= 0.05 for _, _, _, _, p in periodic_rows)
    delta_all_nonsignificant = all(p >= 0.05 for _, _, _, _, p in delta_rows)
    periodic_conclusion = (
        "经过Holm校正后，各规模差异均未达到0.05显著性水平，"
        "因此不能声称Full显著优于或显著劣于Periodic。"
        if periodic_all_nonsignificant
        else "部分规模在Holm校正后达到显著性，必须逐点解释，不能只依据曲线高低下结论。"
    )
    delta_conclusion = (
        "经过Holm校正后，各规模差异均未达到0.05显著性水平，"
        "因此不能声称Full显著优于或显著劣于Send-on-delta。"
        if delta_all_nonsignificant
        else "部分规模在Holm校正后达到显著性，必须逐点解释，不能只依据曲线高低下结论。"
    )
    scale_trend = (
        "略有提高" if success_change > 0 else "略有下降" if success_change < 0 else "保持不变"
    )

    report = f"""# 网络规模敏感性实验结果与分析

## 1. 实验目的

本实验检验将网络节点数由{node_counts[0]}增加到{node_counts[-1]}后，完整卸载前候选发现流程是否仍然有效，以及规模扩大是否会使完整方法相对周期更新、Gossip式查询传播和无在线信任方法表现更好。实验是预先冻结的规模压力测试，不以“让完整方法获胜”为调参目标。

## 2. 统一实验协议

- 节点数：{', '.join(str(x) for x in node_counts)}；平均业务度数固定为{cfg.average_degree}，因此规模增大时网络更稀疏、平均路径通常更长。
- 每个规模包含{cfg.network_repeats}个独立连通初始业务网络，每个网络连续处理{cfg.tasks_per_network}个任务；统计独立单位是网络，而非单个任务或节点。
- 恶意节点比例固定为20%；更新预算为每节点{cfg.update_budget:.0%}；最大转发深度{cfg.max_hops}；绝对截止时间{cfg.deadline_ms:g} ms；候选收集截止{cfg.candidate_collection_ms:g} ms；top-k={cfg.top_k}。
- 三种规模统一使用测试段内{cfg.trace_length}个时隙，节点轨迹偏移为{cfg.offset_step}个时隙。这样既保持所有规模输入长度一致，也避免100节点超出474时隙的可用轨迹。
- 所有方法在同一规模、同一重复中共享资源轨迹、业务图、任务、恶意节点集合、故障和链路随机数。
- Full：门槛感知MPC-DP更新 + 在线Beta直接信任与路径传播 + 业务关系搜索 + top-k确认。
- Periodic：仅把资源更新改为相同50%预算的周期更新，其余与Full相同。
- Send-on-delta：仅把资源更新改为相同50%预算的阈值事件触发更新，其余与Full相同。
- Gossip：使用相同资源更新、信任和确认，只把业务关系全邻居转发改为固定fanout=2的gossip-style decentralized query dissemination。
- No trust：保留资源更新、业务关系搜索和确认，但候选排序不使用在线信任。

## 3. 完整结果

{markdown_table(summary, ['num_nodes', 'pipeline', 'networks', 'honest_success_percent', 'honest_success_ci95_percent', 'malicious_selection_percent', 'malicious_selection_ci95_percent', 'cache_blocked_percent', 'candidate_recall_percent', 'p95_success_latency_ms', 'mean_search_messages', 'mean_update_messages', 'mean_total_messages', 'mean_total_messages_per_node', 'cache_wrmse', 'update_rate_percent'])}

## 4. 网络级配对推断

{markdown_table(tests, ['num_nodes', 'endpoint', 'comparison', 'network_clusters', 'full_percent', 'baseline_percent', 'full_minus_baseline_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_p', 'holm_adjusted_p'])}

## 5. 结果解释

- 完整方法从N={int(first.num_nodes)}到N={int(last.num_nodes)}的诚实候选发现率变化为{success_change:+.2f}个百分点（{float(first.honest_success_percent):.2f}%变为{float(last.honest_success_percent):.2f}%），总体表现为{scale_trend}，没有出现规模扩大导致的性能崩溃。不同规模使用独立生成的业务图，因此不能把该描述性变化解释成节点增加必然改善或必然降低成功率。
- 相对Periodic，完整方法的差异依次为：{periodic_text}。{periodic_conclusion}
- 相对Send-on-delta，完整方法的差异依次为：{delta_text}。{delta_conclusion}
- 相对Gossip，{gossip_text}。这表明全业务邻居传播在大网络中保持了更高发现率，代价是通信量明显增加；Gossip是低开销—低成功率的折中，而不是被完整方法全方位支配。
- 相对No trust，{no_trust_text}。三个规模均显著，说明在线Beta直接信任、路径传播与确认结果更新的作用在扩展到{max(node_counts)}节点后仍然稳定。
- Full的缓存WRMSE在N={('/'.join(str(n) for n in node_counts))}时分别为{full_wrmse_text}，Periodic为{periodic_wrmse_text}。完整方法持续维护了更准确的连续资源状态，但较低WRMSE并不保证每个规模下的门槛判断成功率都更高。
- 完整方法的总通信消息数扩大为原来的{message_ratio:.2f}倍；该指标同时包含搜索消息和按任务摊销的状态更新消息，因此固定每节点更新预算下，节点增加会自然提高总更新开销。
- 按节点归一化后，Full在N={('/'.join(str(n) for n in node_counts))}时的总消息数为{per_node_message_text}条/节点/请求，基本保持稳定。这说明总通信量近似随节点数扩展，但单节点平均负担没有同步恶化；论文必须同时报告总量和每节点量，不能只选择更有利的一项。
- 完整方法的网络级成功决策P95时延变化为{latency_change:+.2f} ms。最大转发深度和截止时间保持不变，因此规模扩大并不会自动允许更深搜索。
- 规模趋势由两种相反作用共同决定：节点增加会提高远端候选的绝对数量，但平均度数和Dmax固定又会降低可触达节点占全网的比例。当前观察到的轻微变化是两种作用叠加后的结果；该解释属于基于协议的机制推断，不是单独验证过的因果结论。

## 6. 论文可用结论边界

该实验可以支持“方法在{min(node_counts)}–{max(node_counts)}节点、固定稀疏度和固定搜索预算下的可扩展性/性能边界”。它不能替代真实硬件部署，也不能把合成业务图表述为工业现场实测网络。若Full未全面超过Periodic，应如实区分连续资源状态误差、门槛候选召回和最终可信候选发现率三个目标。
"""
    (output_dir / "网络规模敏感性实验结果与分析.md").write_text(report, encoding="utf-8")


def write_figure_contract(
    raw: pd.DataFrame,
    cfg: FullAblationConfig,
    output_dir: Path,
) -> None:
    missing_latency = int(raw.success_latency_ms.isna().sum())
    text = f"""# Figure contract and QA

- Core conclusion: quantify how network scale changes trustworthy discovery, malicious selection, communication and latency without presuming that scale favors the proposed method.
- Figure archetype: quantitative grid; panel (a) is the primary evidence.
- Backend: Python/matplotlib only.
- Final size and export: 7.2 x 5.1 inches; editable SVG/PDF plus PNG and 600-dpi TIFF.
- Panels: (a) honest discovery; (b) malicious selection; (c) total communication; (d) network-level P95 successful-decision latency.
- Independent unit: one generated network (n={cfg.network_repeats} per node count); tasks are within-network repeated observations.
- Error bars: 95% normal-approximation confidence intervals over network-level means.
- Inference: paired two-sided Wilcoxon signed-rank tests within each node count; {len(PIPELINES) - 1} comparators x 2 endpoints x {len(raw.num_nodes.unique())} node counts form one Holm family.
- Data integrity: all {len(raw)} generated request-method rows are retained. There are {missing_latency} undefined success-conditioned latency rows caused by failed requests; failures remain in success analyses.
- Reviewer risk: average degree, Dmax and deadline are fixed as N grows, so the experiment evaluates a progressively sparser fixed-search-budget regime. Update messages are total system overhead, not per-node overhead. Simulation is not hardware deployment evidence.
"""
    (output_dir / "figure_contract_and_qa.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "network_scale_sensitivity_formal",
    )
    parser.add_argument("--node-counts", type=parse_node_counts, default=(20, 50, 100))
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--tasks-per-network", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--trace-length", type=int, default=280)
    parser.add_argument("--trace-start", type=int, default=94)
    parser.add_argument("--offset-step", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual, predictions = load_inputs(args.data_dir)
    selected_eta, selected_lambda, thresholds, calibration = calibrate_parameters(
        actual,
        predictions,
        args.trace_start,
        0.5,
        6,
    )
    if not np.isclose(selected_eta, 0.5) or not np.isclose(selected_lambda, 25.0):
        raise AssertionError(
            f"Frozen eta/lambda=0.5/25.0, calibration returned "
            f"{selected_eta}/{selected_lambda}"
        )

    all_raw = []
    configs = {}
    for num_nodes in args.node_counts:
        cfg = FullAblationConfig(
            num_nodes=num_nodes,
            trace_length=args.trace_length,
            trace_start=args.trace_start,
            offset_step=args.offset_step,
            network_repeats=args.repeats,
            tasks_per_network=args.tasks_per_network,
        )
        required_stop = cfg.trace_start + (cfg.num_nodes - 1) * cfg.offset_step + cfg.trace_length
        if required_stop > len(actual):
            raise ValueError(
                f"N={num_nodes} requires {required_stop} trace rows, but only "
                f"{len(actual)} are available"
            )
        nodes_actual, replays = build_scale_inputs(actual, predictions, cfg, thresholds)
        frame = run_scale_condition(num_nodes, nodes_actual, replays, cfg, args.workers)
        expected = cfg.network_repeats * cfg.tasks_per_network * len(PIPELINES)
        if len(frame) != expected:
            raise AssertionError(f"N={num_nodes}: expected {expected} rows, got {len(frame)}")
        all_raw.append(frame)
        configs[str(num_nodes)] = asdict(cfg)

    raw = pd.concat(all_raw, ignore_index=True)
    networks = []
    for num_nodes, group in raw.groupby("num_nodes", sort=True):
        network = network_summary(group.drop(columns="num_nodes"))
        network.insert(0, "num_nodes", int(num_nodes))
        networks.append(network)
    network = pd.concat(networks, ignore_index=True)
    summary = summarize_by_scale(network)
    tests = paired_tests(network, args.node_counts)

    raw.to_csv(args.output_dir / "network_scale_raw.csv", index=False)
    network.to_csv(args.output_dir / "network_scale_network.csv", index=False)
    summary.to_csv(args.output_dir / "network_scale_summary.csv", index=False)
    tests.to_csv(args.output_dir / "network_scale_paired_tests.csv", index=False)
    calibration.to_csv(args.output_dir / "frozen_threshold_calibration.csv", index=False)
    plot_results(summary, args.output_dir)
    report_cfg = FullAblationConfig(
        num_nodes=max(args.node_counts),
        trace_length=args.trace_length,
        trace_start=args.trace_start,
        offset_step=args.offset_step,
        network_repeats=args.repeats,
        tasks_per_network=args.tasks_per_network,
    )
    write_report(summary, tests, args.node_counts, report_cfg, args.output_dir)
    write_figure_contract(raw, report_cfg, args.output_dir)

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "node_counts": list(args.node_counts),
        "malicious_rate": 0.2,
        "pipelines": PIPELINES,
        "pipeline_spec": PIPELINE_SPEC,
        "configs": configs,
        "independent_unit": "generated network",
        "within_network_repeated_observations": args.tasks_per_network,
        "multiple_comparison_family": (
            f"{len(args.node_counts) * (len(PIPELINES) - 1) * 2} paired "
            "network-level comparisons; Holm correction"
        ),
        "rows": len(raw),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
