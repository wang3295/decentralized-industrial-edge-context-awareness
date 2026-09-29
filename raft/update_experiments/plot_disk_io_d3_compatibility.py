"""Create a compact manuscript figure for the disk-I/O d=3 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


METHODS = ("Periodic", "Send-on-delta", "Lyapunov", "MPC-DP")
DISPLAY_NAMES = ("Periodic", "Send-on-\ndelta", "Lyapunov", "MPC-DP")
METHOD_COLORS = {
    "Periodic": "#9A9A9A",
    "Send-on-delta": "#C58A4A",
    "Lyapunov": "#708B9E",
    "MPC-DP": "#2368A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-csv", type=Path, required=True)
    parser.add_argument("--formal-summary-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.set_index("method").loc[list(METHODS)].reset_index()
    if result.isnull().any().any():
        raise ValueError("The plotting input contains missing values")
    return result


def asymmetric_error(
    center: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    return np.vstack([center - lower, upper - center])


def main() -> None:
    args = parse_args()
    configure_style()

    table = ordered(pd.read_csv(args.table_csv))
    formal = ordered(pd.read_csv(args.formal_summary_csv))
    colors = [METHOD_COLORS[method] for method in METHODS]
    x = np.arange(len(METHODS), dtype=float)

    wrmse = formal["wrmse_d3_mean"].to_numpy(dtype=float)
    wrmse_err = asymmetric_error(
        wrmse,
        formal["wrmse_d3_ci95_low"].to_numpy(dtype=float),
        formal["wrmse_d3_ci95_high"].to_numpy(dtype=float),
    )
    disk = formal["diskio_rmse_mean"].to_numpy(dtype=float)
    disk_err = asymmetric_error(
        disk,
        formal["diskio_rmse_ci95_low"].to_numpy(dtype=float),
        formal["diskio_rmse_ci95_high"].to_numpy(dtype=float),
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.45, 3.65),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.18},
    )
    ax_error, ax_latency = axes

    for index, color in enumerate(colors):
        ax_error.errorbar(
            x[index] - 0.10,
            wrmse[index],
            yerr=wrmse_err[:, index : index + 1],
            fmt="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.7,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            capthick=0.9,
            zorder=3,
        )
        ax_error.errorbar(
            x[index] + 0.10,
            disk[index],
            yerr=disk_err[:, index : index + 1],
            fmt="s",
            markersize=4.3,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            capthick=0.9,
            zorder=3,
        )

    ax_error.set_ylabel("Normalized error")
    ax_error.set_xlim(-0.45, len(METHODS) - 0.55)
    ax_error.set_ylim(0.0034, 0.0100)
    ax_error.grid(axis="y", color="#E6E6E6", linewidth=0.65, zorder=0)
    ax_error.tick_params(axis="x", labelbottom=False)
    ax_error.text(
        -0.14,
        1.04,
        "a",
        transform=ax_error.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
    )
    ax_error.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="#444444",
                markerfacecolor="#444444",
                linestyle="none",
                markersize=4.2,
                label="3D WRMSE",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="#444444",
                markerfacecolor="white",
                linestyle="none",
                markersize=4.2,
                label="Disk-I/O RMSE",
            ),
        ],
        loc="upper left",
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.15,
    )

    latency_median = table["industrial_median_decision_ms"].to_numpy(dtype=float)
    latency_p95 = table["industrial_p95_decision_ms"].to_numpy(dtype=float)
    if np.any(latency_median <= 0.0) or np.any(latency_p95 <= 0.0):
        raise ValueError("Decision latencies must be positive for logarithmic plotting")
    for index, color in enumerate(colors):
        ax_latency.vlines(
            x[index],
            latency_median[index],
            latency_p95[index],
            color=color,
            linewidth=1.4,
            zorder=2,
        )
        ax_latency.scatter(
            x[index],
            latency_median[index],
            s=23,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax_latency.scatter(
            x[index],
            latency_p95[index],
            s=50,
            marker="_",
            color=color,
            linewidth=1.25,
            zorder=3,
        )

    ax_latency.set_yscale("log")
    ax_latency.set_ylabel("Decision latency (ms)")
    ax_latency.set_xticks(x, DISPLAY_NAMES)
    ax_latency.set_xlabel("Update policy")
    ax_latency.set_xlim(-0.45, len(METHODS) - 0.55)
    ax_latency.set_ylim(8e-4, 6e-1)
    ax_latency.grid(axis="y", which="major", color="#E6E6E6", linewidth=0.65, zorder=0)
    ax_latency.text(
        -0.14,
        1.04,
        "b",
        transform=ax_latency.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
    )
    ax_latency.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="#444444",
                markerfacecolor="#444444",
                linestyle="none",
                markersize=4.2,
                label="Median",
            ),
            Line2D(
                [0],
                [0],
                marker="_",
                color="#444444",
                linestyle="none",
                markersize=7.0,
                label="P95",
            ),
        ],
        loc="upper left",
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.15,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
