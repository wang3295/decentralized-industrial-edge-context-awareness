from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_joint_discovery_experiment import (
    JointConfig,
    build_cache_replays,
    build_node_replays,
    load_inputs,
    markdown_table,
    run_joint_trials,
)


HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent


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


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def network_cluster_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Treat each independently generated network as one experimental cluster."""
    comparisons = ["last_mpc", "fits_periodic", "fits_delta", "fits_gossip"]
    rows = []
    for workload_idx, workload in enumerate(raw.workload.unique()):
        subset = raw[raw.workload == workload]
        rates = subset.groupby(["repeat", "pipeline"], as_index=False).success.mean()
        proposed = rates[rates.pipeline == "proposed"][["repeat", "success"]].rename(
            columns={"success": "proposed"}
        )
        for comparator_idx, comparator in enumerate(comparisons):
            baseline = rates[rates.pipeline == comparator][["repeat", "success"]].rename(
                columns={"success": "baseline"}
            )
            pair = proposed.merge(baseline, on="repeat", validate="one_to_one")
            difference = 100.0 * (pair.proposed - pair.baseline).to_numpy()
            ci_low, ci_high = bootstrap_mean_ci(
                difference,
                seed=80_000 + workload_idx * 100 + comparator_idx,
            )
            if np.allclose(difference, 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(
                        difference,
                        zero_method="pratt",
                        alternative="two-sided",
                        method="auto",
                    ).pvalue
                )
            rows.append(
                {
                    "workload": workload,
                    "comparison": f"proposed vs {comparator}",
                    "network_clusters": len(pair),
                    "proposed_mean_percent": 100.0 * pair.proposed.mean(),
                    "baseline_mean_percent": 100.0 * pair.baseline.mean(),
                    "paired_difference_pp": difference.mean(),
                    "cluster_bootstrap_ci_low_pp": ci_low,
                    "cluster_bootstrap_ci_high_pp": ci_high,
                    "wilcoxon_network_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def run_gossip_fanout_sensitivity(
    nodes_actual: np.ndarray,
    replays,
    cfg: JointConfig,
    workflow_raw: pd.DataFrame,
    fanouts: Sequence[int] = (1, 2, 3, 4, 5),
) -> pd.DataFrame:
    frames = []
    workflow = workflow_raw[workflow_raw.pipeline == "proposed"].copy()
    workflow["method"] = "Workflow"
    workflow["fanout"] = 0
    frames.append(workflow)
    for fanout in fanouts:
        trial = run_joint_trials(
            nodes_actual,
            replays,
            replace(cfg, gossip_fanout=int(fanout)),
            pipelines=["fits_gossip"],
        )
        trial["method"] = f"Gossip-f{fanout}"
        trial["fanout"] = int(fanout)
        frames.append(trial)
    return pd.concat(frames, ignore_index=True)


def summarize_tradeoff(raw: pd.DataFrame) -> pd.DataFrame:
    summary = raw.groupby(["workload", "method", "fanout"], sort=False).agg(
        tasks=("success", "size"),
        success_rate=("success", "mean"),
        mean_search_messages=("search_messages", "mean"),
        mean_success_latency_ms=("success_latency_ms", "mean"),
        p95_success_latency_ms=("success_latency_ms", lambda x: float(x.quantile(0.95))),
    ).reset_index()
    summary["success_percent"] = 100.0 * summary.success_rate
    summary["success_ci95_percent"] = 100.0 * 1.96 * np.sqrt(
        summary.success_rate * (1.0 - summary.success_rate) / summary.tasks
    )
    return summary


def cluster_tests_workflow_vs_gossip(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for workload_idx, workload in enumerate(raw.workload.unique()):
        subset = raw[raw.workload == workload]
        per_network = subset.groupby(["repeat", "method", "fanout"], as_index=False).agg(
            success_rate=("success", "mean"),
            mean_messages=("search_messages", "mean"),
            p95_latency_ms=("success_latency_ms", lambda x: float(x.quantile(0.95))),
        )
        workflow = per_network[per_network.method == "Workflow"][[
            "repeat", "success_rate", "mean_messages", "p95_latency_ms"
        ]].rename(columns={
            "success_rate": "workflow_success",
            "mean_messages": "workflow_messages",
            "p95_latency_ms": "workflow_p95_latency",
        })
        for fanout in sorted(x for x in subset.fanout.unique() if x > 0):
            gossip = per_network[per_network.fanout == fanout][[
                "repeat", "success_rate", "mean_messages", "p95_latency_ms"
            ]].rename(columns={
                "success_rate": "gossip_success",
                "mean_messages": "gossip_messages",
                "p95_latency_ms": "gossip_p95_latency",
            })
            pair = workflow.merge(gossip, on="repeat", validate="one_to_one")
            success_difference = 100.0 * (
                pair.workflow_success - pair.gossip_success
            ).to_numpy()
            message_difference = (pair.workflow_messages - pair.gossip_messages).to_numpy()
            latency_difference = (
                pair.workflow_p95_latency - pair.gossip_p95_latency
            ).to_numpy()
            ci_low, ci_high = bootstrap_mean_ci(
                success_difference,
                seed=90_000 + workload_idx * 100 + int(fanout),
            )
            p_value = (
                1.0
                if np.allclose(success_difference, 0.0)
                else float(
                    wilcoxon(
                        success_difference,
                        zero_method="pratt",
                        alternative="two-sided",
                        method="auto",
                    ).pvalue
                )
            )
            rows.append(
                {
                    "workload": workload,
                    "fanout": int(fanout),
                    "network_clusters": len(pair),
                    "workflow_minus_gossip_success_pp": success_difference.mean(),
                    "cluster_bootstrap_ci_low_pp": ci_low,
                    "cluster_bootstrap_ci_high_pp": ci_high,
                    "wilcoxon_network_p": p_value,
                    "workflow_minus_gossip_messages": message_difference.mean(),
                    "workflow_minus_gossip_p95_latency_ms": latency_difference.mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_results(
    cluster_tests: pd.DataFrame,
    tradeoff: pd.DataFrame,
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
    comparators = ["last_mpc", "fits_periodic", "fits_delta", "fits_gossip"]
    labels = ["Last-MPC", "Periodic", "Delta", "Gossip-f2"]
    y = np.arange(len(comparators))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), constrained_layout=True)

    for ax, workload, title in [
        (axes[0, 0], "general_feasible", "(a) General: network-level differences"),
        (axes[0, 1], "remote_required", "(b) Remote: network-level differences"),
    ]:
        data = cluster_tests[cluster_tests.workload == workload].set_index("comparison")
        rows = [data.loc[f"proposed vs {name}"] for name in comparators]
        mean = np.array([row.paired_difference_pp for row in rows])
        low = np.array([row.cluster_bootstrap_ci_low_pp for row in rows])
        high = np.array([row.cluster_bootstrap_ci_high_pp for row in rows])
        ax.errorbar(
            mean,
            y,
            xerr=np.vstack([mean - low, high - mean]),
            fmt="o",
            markersize=4,
            capsize=2,
            color="#0072B2",
            linewidth=1.1,
        )
        ax.axvline(0.0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Proposed minus baseline discovery (pp)")
        ax.set_title(title, loc="left", fontweight="bold")

    colors = {1: "#999999", 2: "#009E73", 3: "#E69F00", 4: "#CC79A7", 5: "#D55E00"}
    for ax, workload, title in [
        (axes[1, 0], "general_feasible", "(c) General: success–message trade-off"),
        (axes[1, 1], "remote_required", "(d) Remote: success–message trade-off"),
    ]:
        data = tradeoff[tradeoff.workload == workload]
        workflow = data[data.method == "Workflow"].iloc[0]
        gossip = data[data.fanout > 0].sort_values("fanout")
        ax.plot(
            gossip.mean_search_messages,
            gossip.success_percent,
            color="#666666",
            linewidth=0.9,
            zorder=1,
        )
        annotation_offsets = {
            1: (3, 3),
            2: (3, 3),
            3: (3, 4),
            4: (-34, -11),
            5: (4, -11),
        }
        for row in gossip.itertuples():
            ax.scatter(
                row.mean_search_messages,
                row.success_percent,
                color=colors[int(row.fanout)],
                edgecolor="black",
                linewidth=0.35,
                s=28,
                zorder=2,
            )
            ax.annotate(
                f"f={int(row.fanout)}",
                (row.mean_search_messages, row.success_percent),
                xytext=annotation_offsets[int(row.fanout)],
                textcoords="offset points",
                fontsize=6,
            )
        ax.scatter(
            workflow.mean_search_messages,
            workflow.success_percent,
            marker="*",
            color="#0072B2",
            edgecolor="black",
            linewidth=0.4,
            s=75,
            label="Workflow",
            zorder=3,
        )
        ax.set_xlabel("Search messages per request")
        ax.set_ylabel("Usable-candidate discovery (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5)

    for ax in axes.ravel():
        ax.grid(axis="both", color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    export_figure(fig, output_dir, "cluster_and_gossip_budget_analysis")
    plt.close(fig)


def write_report(
    cluster_tests: pd.DataFrame,
    tradeoff: pd.DataFrame,
    gossip_tests: pd.DataFrame,
    cfg: JointConfig,
    output_dir: Path,
) -> None:
    remote = tradeoff[tradeoff.workload == "remote_required"].copy()
    workflow_remote = remote[remote.method == "Workflow"].iloc[0]
    gossip_remote = remote[remote.fanout > 0].copy()
    gossip_remote["message_gap"] = (
        gossip_remote.mean_search_messages - workflow_remote.mean_search_messages
    ).abs()
    matched = gossip_remote.sort_values(["message_gap", "fanout"]).iloc[0]
    gossip_f3 = gossip_remote[gossip_remote.fanout == 3].iloc[0]
    remote_f3_test = gossip_tests[
        (gossip_tests.workload == "remote_required") & (gossip_tests.fanout == 3)
    ].iloc[0]
    report = f"""# 网络聚类统计与Gossip通信预算实验

## 1. 目的

本实验解决两个审稿风险：第一，不把同一网络中的大量任务错误视为彼此独立，而以{cfg.network_repeats}个动态网络重复作为统计聚类；第二，扫描Gossip fanout=1–5，报告完整发现率–消息开销曲线，而不是只采用一个较弱的Gossip设置。

## 2. 网络聚类统计

每个网络包含每类{cfg.tasks_per_network}个任务。先在网络内计算发现率，再对40个网络的配对差异执行网络级bootstrap 95%置信区间和Wilcoxon配对检验。

{markdown_table(cluster_tests, ['workload', 'comparison', 'network_clusters', 'proposed_mean_percent', 'baseline_mean_percent', 'paired_difference_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_network_p'])}

## 3. Gossip fanout完整权衡

{markdown_table(tradeoff, ['workload', 'method', 'fanout', 'tasks', 'success_percent', 'success_ci95_percent', 'mean_search_messages', 'p95_success_latency_ms'])}

网络级Workflow–Gossip配对结果：

{markdown_table(gossip_tests, ['workload', 'fanout', 'network_clusters', 'workflow_minus_gossip_success_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_network_p', 'workflow_minus_gossip_messages', 'workflow_minus_gossip_p95_latency_ms'])}

## 4. 关键解释

- 远端场景中Workflow为{workflow_remote.success_percent:.2f}%发现率、{workflow_remote.mean_search_messages:.2f}条搜索消息、P95时延{workflow_remote.p95_success_latency_ms:.2f} ms。
- 与Workflow消息数最接近的是Gossip fanout={int(matched.fanout)}：发现率{matched.success_percent:.2f}%，搜索消息{matched.mean_search_messages:.2f}条，P95时延{matched.p95_success_latency_ms:.2f} ms。
- 更有区分度的fanout=3已取得{gossip_f3.success_percent:.2f}%远端发现率，比Workflow仅低{remote_f3_test.workflow_minus_gossip_success_pp:.2f}个百分点；网络级95% CI为[{remote_f3_test.cluster_bootstrap_ci_low_pp:.2f}, {remote_f3_test.cluster_bootstrap_ci_high_pp:.2f}]，Wilcoxon p={remote_f3_test.wilcoxon_network_p:.4f}。它平均使用{gossip_f3.mean_search_messages:.2f}条消息，少于Workflow的{workflow_remote.mean_search_messages:.2f}条。
- fanout提高会使Gossip逐步接近全邻居传播。因此论文应展示完整成功率–消息曲线，并将Workflow优势描述为业务关系约束下的覆盖/时延权衡；如果在相同消息预算下两者收敛，则不能声称Workflow在相同通信量下仍具有明显精度优势。
- 网络聚类统计是正式推断；此前4,800任务级McNemar检验可作为配对任务描述，但不应作为唯一显著性证据。

## 5. 边界

1. Gossip与Workflow使用相同业务图、缓存、故障、截止时间、二次确认和任务样本，仅传播邻居选择规则不同。
2. fanout扫描使用同一正式测试集用于稳健性展示，不用于选择或训练模型参数。
3. 这是离散事件仿真；统计独立单位是动态网络重复，而不是单个任务。
"""
    (output_dir / "网络聚类统计与Gossip预算实验分析.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument(
        "--main-raw",
        type=Path,
        default=HERE / "results" / "candidate_formal" / "joint_discovery_raw.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "cluster_gossip_formal",
    )
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--tasks-per-network", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = JointConfig(network_repeats=args.repeats, tasks_per_network=args.tasks_per_network)
    main_raw = pd.read_csv(args.main_raw)
    expected = cfg.network_repeats * cfg.tasks_per_network
    if not main_raw.groupby(["workload", "pipeline"]).size().eq(expected).all():
        raise ValueError("The supplied main raw file does not match the requested formal design")
    cluster_tests = network_cluster_tests(main_raw)

    actual, predictions = load_inputs(args.data_dir)
    nodes_actual, nodes_predictions = build_node_replays(actual, predictions, cfg)
    replays = build_cache_replays(nodes_actual, nodes_predictions, cfg)
    gossip_raw = run_gossip_fanout_sensitivity(
        nodes_actual,
        replays,
        cfg,
        workflow_raw=main_raw,
    )
    tradeoff = summarize_tradeoff(gossip_raw)
    gossip_tests = cluster_tests_workflow_vs_gossip(gossip_raw)

    cluster_tests.to_csv(output_dir / "network_cluster_tests.csv", index=False)
    gossip_raw.to_csv(output_dir / "gossip_fanout_raw.csv", index=False)
    tradeoff.to_csv(output_dir / "gossip_fanout_summary.csv", index=False)
    gossip_tests.to_csv(output_dir / "gossip_network_cluster_tests.csv", index=False)
    plot_results(cluster_tests, tradeoff, output_dir)
    write_report(cluster_tests, tradeoff, gossip_tests, cfg, output_dir)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(cfg),
        "fanouts": [1, 2, 3, 4, 5],
        "statistical_unit": "network repeat",
        "bootstrap_draws": 20_000,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Network-cluster tests\n", cluster_tests.to_string(index=False))
    print("\nGossip fanout trade-off\n", tradeoff.to_string(index=False))
    print("\nWorkflow-vs-Gossip network tests\n", gossip_tests.to_string(index=False))
    print(f"\nResults saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
