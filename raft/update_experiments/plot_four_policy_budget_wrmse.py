"""Plot the four formal state-update policies across update budgets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


METHODS = ("Periodic", "Send-on-delta", "Lyapunov", "MPC-DP")
COLORS = {
    "Periodic": "#4C78A8",
    "Send-on-delta": "#E68613",
    "Lyapunov": "#2A9D6F",
    "MPC-DP": "#B22222",
}
MARKERS = {
    "Periodic": "s",
    "Send-on-delta": "^",
    "Lyapunov": "D",
    "MPC-DP": "o",
}
LINESTYLES = {
    "Periodic": "--",
    "Send-on-delta": "-.",
    "Lyapunov": (0, (5, 2)),
    "MPC-DP": "-",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    default_input = (
        script_dir
        / "results"
        / "fits_delta_unified_20260914"
        / "industrial_pc"
        / "formal_run"
        / "pareto_summary.csv"
    )
    default_output = (
        project_root
        / "paper_revision_intro_v1"
        / "Decentralized_Adaptive_Context_Awareness_for_Industrial_Edge_Computing"
        / "IOT"
        / "image"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--stem", default="WRMSE under different update budgets"
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"method", "budget", "wrmse_mean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    selected = frame.loc[frame["method"].isin(METHODS), list(required)].copy()
    selected["method"] = pd.Categorical(
        selected["method"], categories=METHODS, ordered=True
    )
    selected = selected.sort_values(["method", "budget"]).reset_index(drop=True)

    counts = selected.groupby("method", observed=True)["budget"].nunique()
    if set(counts.index.astype(str)) != set(METHODS) or not (counts == 5).all():
        raise ValueError(
            "Expected five budget points for each of the four formal policies."
        )
    return selected


def plot(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for method in METHODS:
        data = frame.loc[frame["method"] == method].sort_values("budget")
        is_proposed = method == "MPC-DP"
        ax.plot(
            data["budget"],
            data["wrmse_mean"],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.8 if is_proposed else 1.35,
            markersize=4.3 if is_proposed else 3.8,
            markerfacecolor="white" if is_proposed else COLORS[method],
            markeredgewidth=1.1 if is_proposed else 0.8,
            label=method,
            zorder=4 if is_proposed else 3,
        )

    ax.set_xlabel(r"Update budget ($B$)")
    ax.set_ylabel("WRMSE")
    ax.set_xticks(sorted(frame["budget"].unique()))
    ax.set_xlim(0.075, 0.725)
    ax.set_ylim(0, 3.08)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handlelength=2.2,
        handletextpad=0.5,
        borderaxespad=0.25,
    )
    fig.tight_layout(pad=0.45)

    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    frame.assign(method=frame["method"].astype(str)).to_csv(
        output_dir / f"{stem} source data.csv", index=False
    )


def main() -> None:
    args = parse_args()
    source = load_source(args.input)
    plot(source, args.output_dir, args.stem)
    print(args.output_dir / f"{args.stem}.png")


if __name__ == "__main__":
    main()
