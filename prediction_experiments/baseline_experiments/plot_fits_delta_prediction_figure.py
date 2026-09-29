from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_input = (
        Path(__file__).resolve().parent
        / "results"
        / "prediction_formal_unified_v3"
        / "alibaba_heldout_test_trace_pack_seed2026.npz"
    )
    default_output = (
        root
        / "paper_revision_intro_v1"
        / "Decentralized_Adaptive_Context_Awareness_for_Industrial_Edge_Computing"
        / "IOT"
        / "image"
        / "Comparison of Prediction Results"
    )
    parser = argparse.ArgumentParser(
        description="Plot a traceable FITS-Delta prediction example for the manuscript."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-prefix", type=Path, default=default_output)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument(
        "--forecast-step",
        type=int,
        default=5,
        choices=range(1, 6),
        metavar="{1,...,5}",
    )
    return parser.parse_args()


def load_segment(
    input_path: Path,
    start: int,
    count: int,
    forecast_step: int,
) -> dict[str, np.ndarray]:
    with np.load(input_path, allow_pickle=False) as bundle:
        required = {
            "trace_index",
            "source_row_index_current",
            "actual",
            "true_future",
            "fits_delta_forecast",
        }
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"Missing arrays in {input_path}: {sorted(missing)}")

        actual = np.asarray(bundle["actual"], dtype=float)
        truth = np.asarray(bundle["true_future"], dtype=float)
        forecast = np.asarray(bundle["fits_delta_forecast"], dtype=float)
        trace_index = np.asarray(bundle["trace_index"], dtype=int)
        source_index = np.asarray(bundle["source_row_index_current"], dtype=int)

    if actual.ndim != 2 or actual.shape[1] != 2:
        raise ValueError(f"Expected actual shape (n, 2), got {actual.shape}")
    if truth.shape != (len(actual), 5, 2):
        raise ValueError(f"Expected true_future shape (n, 5, 2), got {truth.shape}")
    if forecast.shape != (len(actual), 6, 2):
        raise ValueError(
            "Expected FITS-Delta output to contain the current state at step 0 "
            f"and five future steps, got {forecast.shape}"
        )
    if not np.allclose(forecast[:, 0, :], actual, atol=1e-6, rtol=0.0):
        raise ValueError("FITS-Delta step 0 does not match the current measured state")
    if start < 0 or count <= 0 or start + count > len(actual):
        raise ValueError(
            f"Requested segment [{start}, {start + count}) is outside 0..{len(actual)}"
        )

    selection = slice(start, start + count)
    horizon_index = forecast_step - 1
    return {
        "trace_index": trace_index[selection],
        "source_row_index_current": source_index[selection],
        "source_row_index_target": source_index[selection] + forecast_step,
        "truth_all": truth[selection, :, :],
        "predicted_all": forecast[selection, 1:6, :],
        "measured": truth[selection, horizon_index, :],
        "predicted": forecast[selection, forecast_step, :],
    }


def write_source_data(output_path: Path, data: dict[str, np.ndarray]) -> None:
    csv_path = output_path.with_name(f"{output_path.name}_source_data.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "heldout_test_window_index",
                "source_row_index_current",
                *[
                    f"{resource}_{kind}_h{horizon}_percent"
                    for resource in ("cpu", "memory")
                    for horizon in range(1, 6)
                    for kind in ("true", "fits_delta")
                ],
            ]
        )
        for row in range(len(data["trace_index"])):
            values = []
            for resource in range(2):
                for horizon in range(5):
                    values.extend(
                        [
                            float(data["truth_all"][row, horizon, resource]),
                            float(data["predicted_all"][row, horizon, resource]),
                        ]
                    )
            writer.writerow(
                [
                    int(data["trace_index"][row]),
                    int(data["source_row_index_current"][row]),
                    *values,
                ]
            )


def make_figure(data: dict[str, np.ndarray], forecast_step: int) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans Mono", "Consolas", "Courier New", "monospace"],
            "font.size": 5.8,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(data["truth_all"]))
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.872), sharex=False)
    channel_names = ("CPU Utilization", "Memory Utilization")

    for channel, (axis, name) in enumerate(zip(axes, channel_names, strict=True)):
        axis.plot(
            x,
            data["truth_all"][:, 0, channel],
            linewidth=0.55,
            alpha=0.7,
            label="True",
            zorder=3,
        )
        for horizon in range(5):
            axis.plot(
                x,
                data["predicted_all"][:, horizon, channel],
                linewidth=0.55,
                alpha=0.7,
                linestyle="--",
                label="Predicted" if horizon == 0 else None,
                zorder=2,
            )
        axis.set_title(name, pad=2.5)
        axis.set_xlabel("Time", labelpad=1.5)
        axis.set_ylabel("Utilization", labelpad=2.0)
        axis.set_xlim(-10, 210)
        axis.set_xticks(np.arange(0, 201, 25))

        values = np.concatenate(
            [
                data["truth_all"][:, 0, channel],
                data["predicted_all"][:, :, channel].reshape(-1),
            ]
        )
        lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
        padding = max((upper - lower) * 0.08, 0.5)
        axis.set_ylim(max(0.0, lower - padding), min(100.0, upper + padding))

    axes[0].legend(loc="upper center", frameon=True)
    axes[1].legend(loc="upper right", frameon=True)
    fig.subplots_adjust(left=0.078, right=0.984, bottom=0.077, top=0.944, hspace=0.40)
    return fig


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")
    png_path = output_prefix.with_suffix(".png")
    tiff_path = output_prefix.with_suffix(".tiff")
    export_options = {"bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(pdf_path, **export_options)
    fig.savefig(svg_path, **export_options)
    fig.savefig(png_path, dpi=600, **export_options)
    rendered = plt.imread(png_path)
    if np.issubdtype(rendered.dtype, np.floating):
        rendered = np.rint(np.clip(rendered, 0.0, 1.0) * 255).astype(np.uint8)
    tifffile.imwrite(
        tiff_path,
        rendered,
        photometric="rgb",
        compression="deflate",
        resolution=(600, 600),
        resolutionunit="INCH",
    )


def main() -> None:
    args = parse_args()
    data = load_segment(args.input, args.start, args.count, args.forecast_step)
    write_source_data(args.output_prefix, data)
    figure = make_figure(data, args.forecast_step)
    save_figure(figure, args.output_prefix)
    plt.close(figure)

    residual = data["predicted"] - data["measured"]
    rmse = np.sqrt(np.mean(np.square(residual), axis=0))
    mae = np.mean(np.abs(residual), axis=0)
    print(
        f"Exported {args.output_prefix}.* from {len(data['trace_index'])} held-out windows; "
        f"CPU RMSE/MAE={rmse[0]:.4f}/{mae[0]:.4f}, "
        f"memory RMSE/MAE={rmse[1]:.4f}/{mae[1]:.4f}."
    )


if __name__ == "__main__":
    main()
