from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROPOSED_COLOR = "#0072B2"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.6,
            "axes.titlesize": 7.3,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.45,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    x_label: str,
    letter: str,
    color: str,
) -> None:
    levels = data["level"].to_numpy(dtype=float)
    values = data["honest_success_percent"].to_numpy(dtype=float)
    ci_low = data["honest_success_percent_ci_low"].to_numpy(dtype=float)
    ci_high = data["honest_success_percent_ci_high"].to_numpy(dtype=float)

    ax.fill_between(levels, ci_low, ci_high, color=color, alpha=0.15, linewidth=0)
    ax.plot(
        levels,
        values,
        color=color,
        marker="o",
        markersize=3.8,
        markeredgewidth=0.6,
        zorder=3,
    )
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", pad=3)
    ax.set_xlabel(x_label, labelpad=2)
    ax.set_xticks(levels)
    if np.allclose(levels, np.round(levels)):
        ax.set_xticklabels([str(int(value)) for value in levels])
    ax.set_ylim(20, 92)
    ax.set_yticks([20, 40, 60, 80])
    ax.tick_params(axis="both", width=0.7, length=2.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure(summary_csv: Path, output_base: Path) -> None:
    configure_style()
    frame = pd.read_csv(summary_csv)
    frame = frame.loc[frame["experiment"].eq("hyperparameter")].copy()

    specifications = [
        ("deadline_ms", "Response deadline", "Deadline (ms)", "a", PROPOSED_COLOR),
        ("max_hops", "Search depth", "Search depth (hops)", "b", PROPOSED_COLOR),
        ("top_k", "Reconfirmation", "Candidates selected, $k$", "c", PROPOSED_COLOR),
        ("gossip_fanout", "Gossip fanout", "Fanout", "d", PROPOSED_COLOR),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.25), sharey=True)
    for ax, (factor, title, x_label, letter, color) in zip(axes.flat, specifications):
        data = frame.loc[frame["factor"].eq(factor)].sort_values("level")
        if data.empty:
            raise ValueError(f"No rows found for factor {factor!r}")
        draw_panel(ax, data, title, x_label, letter, color)

    fig.supylabel("Trustworthy candidate-discovery rate (%)", fontsize=6.8, x=0.02)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.13, top=0.96, wspace=0.16, hspace=0.36)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("output_base", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_figure(args.summary_csv, args.output_base)
