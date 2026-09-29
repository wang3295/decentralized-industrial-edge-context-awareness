from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep text editable in SVG/PDF and use a journal-safe sans-serif stack.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


METHODS = ["Periodic", "Send-on-delta", "Lyapunov", "MPC-DP"]
COLORS = {
    "Periodic": "#4C78A8",
    "Send-on-delta": "#F28E2B",
    "Lyapunov": "#59A14F",
    "MPC-DP": "#B64342",
}
MARKERS = {
    "Periodic": "s",
    "Send-on-delta": "^",
    "Lyapunov": "D",
    "MPC-DP": "X",
}
LINESTYLES = {
    "Periodic": "--",
    "Send-on-delta": "-.",
    "Lyapunov": (0, (5, 2)),
    "MPC-DP": "-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_and_validate(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pareto = pd.read_csv(input_dir / "pareto_summary.csv")
    main = pd.read_csv(input_dir / "main_B0.5_summary.csv")
    pareto = pareto[pareto["method"].isin(METHODS)].copy()
    main = main[main["method"].isin(METHODS)].copy()

    expected_budgets = {0.1, 0.2, 0.3, 0.5, 0.7}
    if set(np.round(pareto["budget"], 10)) != expected_budgets:
        raise ValueError("The Pareto source does not contain all five planned budgets")
    if set(main["method"]) != set(METHODS) or len(main) != len(METHODS):
        raise ValueError("The B=0.5 source does not contain exactly four formal methods")
    if not np.allclose(main["update_count_mean"], main["update_count_mean"].iloc[0]):
        raise ValueError("B=0.5 methods do not have identical update counts")
    if not np.allclose(main["update_rate_mean"], main["update_rate_mean"].iloc[0]):
        raise ValueError("B=0.5 methods do not have identical update rates")
    if not np.allclose(main["budget_violation_count_mean"], 0):
        raise ValueError("At least one B=0.5 method violates the communication budget")
    return pareto, main


def write_source_and_table(main: pd.DataFrame, pareto: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    order = {name: index for index, name in enumerate(METHODS)}
    main = main.assign(_order=main["method"].map(order)).sort_values("_order")

    table = pd.DataFrame(
        {
            "Method": main["method"],
            "WRMSE": main["wrmse_mean"],
            "CPU RMSE": main["cpu_rmse_mean"],
            "Memory RMSE": main["memory_rmse_mean"],
            "Updates": main["update_count_mean"].round().astype(int),
            "Actual update rate": main["update_rate_mean"],
            "Mean AoI": main["mean_aoi_mean"],
            "P95 AoI": main["p95_aoi_mean"],
            "Median decision time (ms)": main["decision_median_ms_mean"],
            "P95 decision time (ms)": main["decision_p95_ms_mean"],
            "Budget violations": main["budget_violation_count_mean"].round().astype(int),
        }
    )
    table.to_csv(output_dir / "update_policy_main_table_B0.5.csv", index=False)
    pareto.sort_values(["budget", "method"]).to_csv(
        output_dir / "update_policy_main_figure_source_data.csv", index=False
    )

    rows = []
    for item in table.to_dict(orient="records"):
        method = f"**{item['Method']}**" if item["Method"] == "MPC-DP" else item["Method"]
        wrmse = f"**{item['WRMSE']:.4f}**" if item["Method"] == "MPC-DP" else f"{item['WRMSE']:.4f}"
        rows.append(
            "| "
            + " | ".join(
                [
                    method,
                    wrmse,
                    f"{item['CPU RMSE']:.4f}",
                    f"{item['Memory RMSE']:.4f}",
                    f"{item['Updates']}/{item['Actual update rate']:.4f}",
                    f"{item['Mean AoI']:.3f}/{item['P95 AoI']:.3f}",
                    f"{item['Median decision time (ms)']:.5f}/{item['P95 decision time (ms)']:.5f}",
                    str(item["Budget violations"]),
                ]
            )
            + " |"
        )

    markdown = "\n".join(
        [
            "# 状态更新策略主表（B=0.5）",
            "",
            "所有方法在379个测试时隙中均执行189次逻辑更新，实际更新率为0.4987。",
            "",
            "| 方法 | WRMSE↓ | CPU RMSE↓ | Memory RMSE↓ | 更新数/率↓ | 平均/P95 AoI↓ | 决策中位/P95（ms）↓ | 预算违反↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "说明：决策时延仅包含更新策略本身，不包含FITS推理、文件读取和网络传输。",
            "逻辑更新数是单条状态发送决策数；在多邻居网络中的应用层消息数还需乘以实际接收邻居数。",
        ]
    )
    (output_dir / "状态更新策略主表_B0.5.md").write_text(markdown, encoding="utf-8")


def draw_main_figure(pareto: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    for method in METHODS:
        data = pareto[pareto["method"] == method].sort_values("update_rate_mean")
        x = 100.0 * data["update_rate_mean"].to_numpy()
        y = data["wrmse_mean"].to_numpy()
        ax.plot(
            x,
            y,
            label=method,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.45 if method == "MPC-DP" else 1.15,
            markersize=4.5 if method == "MPC-DP" else 3.8,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=3 if method == "MPC-DP" else 2,
        )

    main_rate = 100.0 * pareto.loc[np.isclose(pareto["budget"], 0.5), "update_rate_mean"].iloc[0]
    ax.axvline(main_rate, color="#767676", linestyle=":", linewidth=0.9, zorder=1)
    ax.text(
        main_rate + 1.2,
        0.10,
        r"Main budget $B=0.5$",
        color="#606060",
        fontsize=6.7,
        rotation=90,
        va="bottom",
        ha="left",
    )
    ax.set_xlabel("Actual update rate (%)")
    ax.set_ylabel("Cache-state WRMSE")
    ax.set_xlim(6, 73)
    ax.set_ylim(0, 3.55)
    ax.set_xticks([10, 20, 30, 50, 70])
    ax.grid(axis="both", color="#D8D8D8", linewidth=0.55, alpha=0.7)
    ax.legend(loc="upper right", ncol=2, handlelength=2.2, columnspacing=0.9)
    fig.tight_layout(pad=0.7)

    stem = output_dir / "update_policy_wrmse_update_rate_main"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    caption = """# Figure legend

**Fig. X | Accuracy-communication trade-off of resource-state update policies.**
Cache-state weighted root mean squared error (WRMSE) is plotted against the actual update rate on the held-out 379-slot test sequence. CPU and memory errors receive equal weights (0.5/0.5). At every budget, Periodic, Send-on-delta, Lyapunov and MPC-DP use the same exact number of logical updates; hyperparameters are selected only on the preceding validation segment. MPC-DP achieves the lowest WRMSE at B=0.2, 0.5 and 0.7, whereas Send-on-delta is best at B=0.1 and 0.3. The vertical dotted line marks the prespecified main budget B=0.5. Lower values indicate more accurate remote resource caches.
"""
    (output_dir / "update_policy_main_figure_legend.md").write_text(caption, encoding="utf-8")


def write_qa(output_dir: Path, pareto: pd.DataFrame) -> None:
    notes = "\n".join(
        [
            "# 主图数据与QA说明",
            "",
            "- 核心结论：在精确匹配的更新预算下，MPC-DP在B=0.2、0.5和0.7取得最低WRMSE，但并非所有预算均最优。",
            "- 图形类型：单栏定量折线图；主证据为WRMSE—实际更新率权衡。",
            "- 数据完整性：四种正式更新策略、五个预设预算点全部展示，共20个聚合观测点。",
            "- 对照规则：同一预算下四种方法的实际更新数完全一致，且预算违反均为0。",
            "- 调参边界：Send-on-delta阈值、Lyapunov V0和MPC-DP eta仅由验证段选择。",
            "- 不确定性：四种方法在固定轨迹和固定参数下是确定性运行，因此该图不绘制伪造的种子误差条。",
            "- 审稿风险：不能据图宣称MPC-DP在所有预算下全面优于Send-on-delta。",
            f"- 完整观测数：{len(pareto)}。",
        ]
    )
    (output_dir / "update_policy_main_figure_QA.md").write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()
    pareto, main_frame = load_and_validate(args.input_dir)
    write_source_and_table(main_frame, pareto, args.output_dir)
    draw_main_figure(pareto, args.output_dir)
    write_qa(args.output_dir, pareto)
    print(f"Paper outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
