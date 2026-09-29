"""Fair comparison of resource-state update strategies.

The legacy scripts under ``raft/`` are intentionally left unchanged.  This
entry point fixes the metric and budget inconsistencies and compares:

* Random
* Periodic
* Send-on-delta
* Lyapunov
* MPC-DP (proposed)

All strategies are evaluated with the same exact number of update messages.
The first 20% of the trace is used only for hyper-parameter calibration, and
the remaining 80% is the held-out test segment.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAFT_ROOT = Path(__file__).resolve().parents[1]
if str(RAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(RAFT_ROOT))

from edgebench_core.update_policies import (  # noqa: E402
    DEFAULT_WEIGHTS,
    ExactTraceBudgetGuard as BudgetGuard,
    mpc_dp_first_action,
    normalize_weights,
    weighted_squared_error,
)


METHODS = ("Random", "Periodic", "Send-on-delta", "Lyapunov", "MPC-DP")
WEIGHTS = DEFAULT_WEIGHTS.copy()


@dataclass
class RunResult:
    method: str
    budget: float
    seed: int
    wrmse: float
    balanced_wrmse: float
    cpu_rmse: float
    memory_rmse: float
    update_count: int
    update_rate: float
    successful_update_count: int
    effective_update_rate: float
    failed_update_count: int
    loss_probability: float
    budget_violation_count: int
    blocked_proposals: int
    mean_aoi: float
    p95_aoi: float
    max_aoi: int
    decision_median_ms: float
    decision_p95_ms: float
    decision_mean_ms: float
    parameter: float | None
    horizon: int
    cpu_weight: float
    memory_weight: float


def load_inputs(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    actual = pd.read_csv(data_dir / "value.csv", header=None).to_numpy(dtype=float)
    cpu = pd.read_csv(data_dir / "cpu_predictions.csv").to_numpy(dtype=float)
    memory = pd.read_csv(data_dir / "memory_predictions.csv").to_numpy(dtype=float)
    if actual.ndim != 2 or actual.shape[1] != 2:
        raise ValueError(f"Expected value.csv with two columns, got {actual.shape}")
    if cpu.shape != memory.shape:
        raise ValueError(f"CPU and memory prediction shapes differ: {cpu.shape}, {memory.shape}")
    if cpu.shape[0] != actual.shape[0]:
        raise ValueError(f"Prediction/actual lengths differ: {cpu.shape[0]}, {actual.shape[0]}")
    predictions = np.stack([cpu, memory], axis=2)
    max_step0_error = float(np.max(np.abs(predictions[:, 0, :] - actual)))
    if max_step0_error > 1e-5:
        raise ValueError(f"step_0 is not aligned with actual values; max error={max_step0_error}")
    return actual, predictions


def run_strategy(
    method: str,
    actual: np.ndarray,
    predictions: np.ndarray,
    initial_cache: np.ndarray,
    budget: float,
    seed: int = 40,
    parameter: float | None = None,
    horizon: int = 6,
    lyapunov_power: float = 0.5,
    weights: np.ndarray | None = None,
    loss_probability: float = 0.0,
) -> RunResult:
    active_weights = normalize_weights(weights)
    if not 0.0 <= loss_probability <= 1.0:
        raise ValueError("loss_probability must be in [0, 1]")
    length = len(actual)
    guard = BudgetGuard(length, budget)
    rng = np.random.default_rng(seed)
    loss_rng = np.random.default_rng(seed + 1_000_003)
    cache = np.asarray(initial_cache, dtype=float).copy()
    queue = 0.0
    errors: list[np.ndarray] = []
    aois: list[int] = []
    latencies_ns: list[int] = []
    age = 0
    successful_updates = 0
    failed_updates = 0

    for t in range(length):
        start_ns = time.perf_counter_ns()
        if method == "Random":
            remaining_slots = length - t
            remaining_messages = guard.target - guard.used
            proposed = bool(rng.random() < remaining_messages / remaining_slots)
        elif method == "Periodic":
            before = int(math.floor(t * guard.target / length))
            after = int(math.floor((t + 1) * guard.target / length))
            proposed = after > before
        elif method == "Send-on-delta":
            if parameter is None:
                raise ValueError("Send-on-delta requires a threshold")
            delta = math.sqrt(weighted_squared_error(actual[t], cache, active_weights))
            proposed = delta > parameter
        elif method == "Lyapunov":
            if parameter is None:
                raise ValueError("Lyapunov requires V0")
            error_if_silent = weighted_squared_error(actual[t], cache, active_weights)
            vt = float(parameter) * (t + 2) ** lyapunov_power
            cost_silent = vt * error_if_silent + queue * (0.0 - budget)
            cost_send = queue * (1.0 - budget)
            proposed = cost_send < cost_silent
        elif method == "MPC-DP":
            if parameter is None:
                raise ValueError("MPC-DP requires eta")
            forecast = predictions[t, : min(horizon, predictions.shape[1]), :]
            local_budget = min(
                guard.target - guard.used,
                int(math.ceil(budget * len(forecast))),
            )
            proposed = mpc_dp_first_action(forecast, cache, local_budget, parameter, active_weights)
        else:
            raise ValueError(f"Unknown strategy: {method}")

        action = guard.apply(proposed, t)
        if method == "Lyapunov":
            queue = max(queue + float(action) - budget, 0.0)
        if action:
            lost = bool(loss_probability > 0.0 and loss_rng.random() < loss_probability)
            if lost:
                failed_updates += 1
                age += 1
            else:
                successful_updates += 1
                cache = actual[t].copy()
                age = 0
        else:
            age += 1
        error = actual[t] - cache
        end_ns = time.perf_counter_ns()
        errors.append(error)
        aois.append(age)
        latencies_ns.append(end_ns - start_ns)

    if guard.used != guard.target:
        raise AssertionError(f"{method}: used={guard.used}, target={guard.target}")
    error_array = np.asarray(errors, dtype=float)
    per_slot_weighted_mse = np.sum(active_weights * np.square(error_array), axis=1)
    balanced_mse = np.sum(WEIGHTS * np.square(error_array), axis=1)
    latency_ms = np.asarray(latencies_ns, dtype=float) / 1e6
    return RunResult(
        method=method,
        budget=float(budget),
        seed=int(seed),
        wrmse=float(np.sqrt(np.mean(per_slot_weighted_mse))),
        balanced_wrmse=float(np.sqrt(np.mean(balanced_mse))),
        cpu_rmse=float(np.sqrt(np.mean(np.square(error_array[:, 0])))),
        memory_rmse=float(np.sqrt(np.mean(np.square(error_array[:, 1])))),
        update_count=int(guard.used),
        update_rate=float(guard.used / length),
        successful_update_count=int(successful_updates),
        effective_update_rate=float(successful_updates / length),
        failed_update_count=int(failed_updates),
        loss_probability=float(loss_probability),
        budget_violation_count=int(guard.violations),
        blocked_proposals=int(guard.blocked),
        mean_aoi=float(np.mean(aois)),
        p95_aoi=float(np.percentile(aois, 95)),
        max_aoi=int(np.max(aois)),
        decision_median_ms=float(np.median(latency_ms)),
        decision_p95_ms=float(np.percentile(latency_ms, 95)),
        decision_mean_ms=float(np.mean(latency_ms)),
        parameter=None if parameter is None else float(parameter),
        horizon=int(horizon),
        cpu_weight=float(active_weights[0]),
        memory_weight=float(active_weights[1]),
    )


def select_parameter(
    method: str,
    candidates: Iterable[float],
    actual: np.ndarray,
    predictions: np.ndarray,
    initial_cache: np.ndarray,
    budget: float,
    horizon: int,
    weights: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for candidate in candidates:
        result = run_strategy(
            method,
            actual,
            predictions,
            initial_cache,
            budget,
            seed=40,
            parameter=float(candidate),
            horizon=horizon,
            weights=weights,
        )
        rows.append(asdict(result))
    frame = pd.DataFrame(rows).sort_values(["wrmse", "mean_aoi", "parameter"])
    return float(frame.iloc[0]["parameter"]), frame


def delta_candidates(actual: np.ndarray) -> np.ndarray:
    changes = np.sqrt(np.sum(WEIGHTS * np.square(np.diff(actual, axis=0)), axis=1))
    values = np.quantile(changes, np.linspace(0.0, 1.0, 41))
    return np.unique(np.concatenate(([0.0], values, [float(np.max(changes) + 1e-6)])))


def aggregate_runs(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "wrmse",
        "balanced_wrmse",
        "cpu_rmse",
        "memory_rmse",
        "update_count",
        "update_rate",
        "successful_update_count",
        "effective_update_rate",
        "failed_update_count",
        "budget_violation_count",
        "blocked_proposals",
        "mean_aoi",
        "p95_aoi",
        "max_aoi",
        "decision_median_ms",
        "decision_p95_ms",
        "decision_mean_ms",
    ]
    grouped = raw.groupby(["method", "budget"], sort=False)[metric_columns]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    return means.join(stds).reset_index()


def make_oracle_predictions(actual: np.ndarray, horizon: int) -> np.ndarray:
    oracle = np.empty((len(actual), horizon, actual.shape[1]), dtype=float)
    for t in range(len(actual)):
        for h in range(horizon):
            oracle[t, h] = actual[min(t + h, len(actual) - 1)]
    return oracle


def make_last_value_predictions(actual: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(actual[:, None, :], horizon, axis=1)


def forecast_metrics(
    actual: np.ndarray,
    predictions: np.ndarray,
    horizon: int,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    errors = []
    for h in range(1, horizon):
        valid = len(actual) - h
        if valid > 0:
            errors.append(predictions[:valid, h, :] - actual[h:, :])
    if not errors:
        return {"forecast_wrmse": float("nan"), "forecast_cpu_rmse": float("nan"), "forecast_memory_rmse": float("nan")}
    error = np.vstack(errors)
    active_weights = normalize_weights(weights)
    return {
        "forecast_wrmse": float(np.sqrt(np.mean(np.sum(active_weights * np.square(error), axis=1)))),
        "forecast_cpu_rmse": float(np.sqrt(np.mean(np.square(error[:, 0])))),
        "forecast_memory_rmse": float(np.sqrt(np.mean(np.square(error[:, 1])))),
    }


def run_prediction_ablation(
    actual: np.ndarray,
    fits_predictions: np.ndarray,
    initial_cache: np.ndarray,
    budget: float,
    eta: float,
    horizon: int,
) -> pd.DataFrame:
    variants = [
        ("Oracle future", make_oracle_predictions(actual, horizon), horizon, True),
        ("FITS forecast", fits_predictions, horizon, True),
        ("Last-value forecast", make_last_value_predictions(actual, horizon), horizon, True),
        ("Current-only (H=1)", make_last_value_predictions(actual, 1), 1, False),
    ]
    rows = []
    for label, prediction, variant_horizon, has_future in variants:
        result = asdict(
            run_strategy(
                "MPC-DP",
                actual,
                prediction,
                initial_cache,
                budget,
                parameter=eta,
                horizon=variant_horizon,
            )
        )
        result["variant"] = label
        result.update(
            forecast_metrics(actual, prediction, variant_horizon)
            if has_future
            else {"forecast_wrmse": float("nan"), "forecast_cpu_rmse": float("nan"), "forecast_memory_rmse": float("nan")}
        )
        rows.append(result)
    return pd.DataFrame(rows)


def run_calibrated_prediction_ablation(
    val_actual: np.ndarray,
    val_fits: np.ndarray,
    val_initial: np.ndarray,
    test_actual: np.ndarray,
    test_fits: np.ndarray,
    test_initial: np.ndarray,
    budget: float,
    eta_candidates: Iterable[float],
    horizon: int,
) -> pd.DataFrame:
    variants = [
        (
            "Perfect-forecast MPC",
            make_oracle_predictions(val_actual, horizon),
            make_oracle_predictions(test_actual, horizon),
            horizon,
            True,
        ),
        ("FITS forecast", val_fits, test_fits, horizon, True),
        (
            "Last-value forecast",
            make_last_value_predictions(val_actual, horizon),
            make_last_value_predictions(test_actual, horizon),
            horizon,
            True,
        ),
        (
            "Current-only (H=1)",
            make_last_value_predictions(val_actual, 1),
            make_last_value_predictions(test_actual, 1),
            1,
            False,
        ),
    ]
    rows = []
    for label, val_prediction, test_prediction, variant_horizon, has_future in variants:
        selected_eta, calibration = select_parameter(
            "MPC-DP",
            eta_candidates,
            val_actual,
            val_prediction,
            val_initial,
            budget,
            variant_horizon,
        )
        result = asdict(
            run_strategy(
                "MPC-DP",
                test_actual,
                test_prediction,
                test_initial,
                budget,
                parameter=selected_eta,
                horizon=variant_horizon,
            )
        )
        result["variant"] = label
        result["selected_eta"] = selected_eta
        result["validation_wrmse"] = float(calibration.iloc[0].wrmse)
        result.update(
            forecast_metrics(test_actual, test_prediction, variant_horizon)
            if has_future
            else {"forecast_wrmse": float("nan"), "forecast_cpu_rmse": float("nan"), "forecast_memory_rmse": float("nan")}
        )
        rows.append(result)
    return pd.DataFrame(rows)


def run_weight_sensitivity(
    actual: np.ndarray,
    predictions: np.ndarray,
    initial_cache: np.ndarray,
    budget: float,
    eta: float,
    horizon: int,
) -> pd.DataFrame:
    rows = []
    for cpu_weight in (0.2, 0.5, 0.8):
        weights = np.asarray([cpu_weight, 1.0 - cpu_weight], dtype=float)
        rows.append(
            asdict(
                run_strategy(
                    "MPC-DP",
                    actual,
                    predictions,
                    initial_cache,
                    budget,
                    parameter=eta,
                    horizon=horizon,
                    weights=weights,
                )
            )
        )
    return pd.DataFrame(rows)


def run_loss_robustness(
    actual: np.ndarray,
    predictions: np.ndarray,
    initial_cache: np.ndarray,
    budget: float,
    delta: float,
    lyapunov_v0: float,
    eta: float,
    horizon: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameters = {
        "Random": None,
        "Periodic": None,
        "Send-on-delta": delta,
        "Lyapunov": lyapunov_v0,
        "MPC-DP": eta,
    }
    rows = []
    for loss_probability in (0.0, 0.05, 0.10, 0.20):
        for method in METHODS:
            for seed in range(40, 40 + repeats):
                rows.append(
                    asdict(
                        run_strategy(
                            method,
                            actual,
                            predictions,
                            initial_cache,
                            budget,
                            seed=seed,
                            parameter=parameters[method],
                            horizon=horizon,
                            loss_probability=loss_probability,
                        )
                    )
                )
    raw = pd.DataFrame(rows)
    metrics = [
        "wrmse",
        "balanced_wrmse",
        "cpu_rmse",
        "memory_rmse",
        "update_count",
        "update_rate",
        "successful_update_count",
        "effective_update_rate",
        "failed_update_count",
        "mean_aoi",
        "p95_aoi",
        "max_aoi",
    ]
    grouped = raw.groupby(["method", "loss_probability"], sort=False)[metrics]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).fillna(0.0).add_suffix("_std")).reset_index()
    return raw, summary


def save_plots(
    pareto: pd.DataFrame,
    main: pd.DataFrame,
    eta: pd.DataFrame,
    ablation: pd.DataFrame,
    output: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    def export(fig: plt.Figure, stem: str) -> None:
        fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{stem}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")

    colors = {
        "Random": "#8c8c8c",
        "Periodic": "#4c78a8",
        "Send-on-delta": "#f58518",
        "Lyapunov": "#54a24b",
        "MPC-DP": "#b22222",
    }
    markers = {"Random": "o", "Periodic": "s", "Send-on-delta": "^", "Lyapunov": "D", "MPC-DP": "X"}
    linestyles = {"Random": ":", "Periodic": "--", "Send-on-delta": "-.", "Lyapunov": (0, (5, 2)), "MPC-DP": "-"}
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    for method in METHODS:
        data = pareto[pareto.method == method].sort_values("update_rate_mean")
        ax.plot(
            data.update_rate_mean,
            data.wrmse_mean,
            marker=markers[method],
            linestyle=linestyles[method],
            markersize=4.2,
            linewidth=1.4,
            label=method,
            color=colors[method],
        )
        if method == "Random":
            ax.fill_between(
                data.update_rate_mean,
                data.wrmse_mean - data.wrmse_std,
                data.wrmse_mean + data.wrmse_std,
                alpha=0.15,
                color=colors[method],
            )
    ax.set_xlabel("Actual update rate")
    ax.set_ylabel("WRMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    export(fig, "wrmse_update_rate_pareto")
    plt.close(fig)

    main = main.set_index("method").loc[list(METHODS)].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(main))
    hatches = ["//", "..", "xx", "\\\\", ""]
    bars0 = axes[0].bar(
        x,
        main.wrmse_mean,
        yerr=main.wrmse_std,
        capsize=2,
        color=[colors[m] for m in main.method],
        edgecolor="black",
        linewidth=0.4,
    )
    for bar, hatch in zip(bars0, hatches):
        bar.set_hatch(hatch)
    axes[0].set_xticks(x, main.method, rotation=25, ha="right")
    axes[0].set_ylabel("WRMSE (B = 0.5)")
    axes[0].grid(axis="y", alpha=0.25)
    bars1 = axes[1].bar(
        x,
        main.mean_aoi_mean,
        yerr=main.mean_aoi_std,
        capsize=2,
        color=[colors[m] for m in main.method],
        edgecolor="black",
        linewidth=0.4,
    )
    for bar, hatch in zip(bars1, hatches):
        bar.set_hatch(hatch)
    axes[1].set_xticks(x, main.method, rotation=25, ha="right")
    axes[1].set_ylabel("Mean AoI (slots)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    export(fig, "main_budget_comparison")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.5))
    ax1.plot(eta.parameter, eta.wrmse, marker="o", linewidth=1.4, color=colors["MPC-DP"], label="WRMSE")
    ax1.set_xlabel("eta")
    ax1.set_ylabel("WRMSE", color=colors["MPC-DP"])
    ax1.tick_params(axis="y", labelcolor=colors["MPC-DP"])
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(eta.parameter, eta.mean_aoi, marker="s", linestyle="--", linewidth=1.2, color="#4c78a8", label="Mean AoI")
    ax2.set_ylabel("Mean AoI (slots)", color="#4c78a8")
    ax2.tick_params(axis="y", labelcolor="#4c78a8")
    ax1.axvline(1.0, color="#666666", linestyle=":", linewidth=1.0)
    ax1.text(1.03, ax1.get_ylim()[1] * 0.96, "validation-selected", va="top", color="#555555")
    fig.tight_layout()
    export(fig, "eta_sensitivity")
    plt.close(fig)

    labels = ["Perfect", "FITS", "Last-value", "Current-only"]
    ablation = ablation.copy()
    ablation["short_label"] = labels
    ablation_colors = ["#9a9a9a", colors["MPC-DP"], "#4c78a8", "#f58518"]
    ablation_hatches = ["//", "", "..", "xx"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    x = np.arange(len(ablation))
    bars = axes[0].bar(x, ablation.wrmse, color=ablation_colors, edgecolor="black", linewidth=0.4)
    for bar, hatch in zip(bars, ablation_hatches):
        bar.set_hatch(hatch)
    axes[0].set_xticks(x, ablation.short_label, rotation=22, ha="right")
    axes[0].set_ylabel("Cache-state WRMSE")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, ablation.wrmse):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6)

    width = 0.36
    axes[1].bar(x - width / 2, ablation.cpu_rmse, width, label="CPU", color="#4c78a8", edgecolor="black", linewidth=0.4)
    axes[1].bar(x + width / 2, ablation.memory_rmse, width, label="Memory", color="#f58518", edgecolor="black", linewidth=0.4, hatch="//")
    axes[1].set_xticks(x, ablation.short_label, rotation=22, ha="right")
    axes[1].set_ylabel("Resource RMSE")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    export(fig, "prediction_input_ablation")
    plt.close(fig)

    qa_notes = """# Figure contract and QA notes

- Core conclusion: under identical communication budgets, MPC-DP reduces state-cache WRMSE at B=0.2, 0.5, and 0.7; the prediction-input ablation tests whether future information contributes beyond current-only scheduling.
- Archetype: quantitative comparison grid.
- Backend: Python/matplotlib only.
- Target/output: IEEE double-column candidate figures, 7.2-inch width; editable SVG/PDF plus 600-dpi TIFF and PNG preview.
- Hero evidence: WRMSE-update-rate Pareto curve.
- Validation evidence: B=0.5 WRMSE/AoI comparison.
- Robustness evidence: eta sensitivity with validation-selected eta explicitly marked, plus perfect-forecast/FITS/last-value/current-only prediction-input ablation; each input variant selects eta on the validation segment only.
- Train/validation/test protocol: first 20% calibration and last 80% held-out testing.
- Variability: Random uses 30 seeds (mean and standard deviation); deterministic methods have no seed-derived error bar.
- Metric: WRMSE = sqrt(mean_t(sum_j(w_j*(x_tj-z_tj)^2))), with CPU/memory weights 0.5/0.5.
- Reviewer risk: do not claim universal superiority; Send-on-delta wins at two budgets, and timing is from a desktop x86 CPU rather than edge hardware.
- Source data: pareto_summary.csv, main_B0.5_summary.csv, eta_sensitivity.csv, prediction_ablation.csv.
"""
    (output / "figure_contract_and_qa.md").write_text(qa_notes, encoding="utf-8")


def save_robustness_plots(weight_frame: pd.DataFrame, loss_frame: pd.DataFrame, output: Path) -> None:
    def export(fig: plt.Figure, stem: str) -> None:
        fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{stem}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    x = weight_frame.cpu_weight
    axes[0].plot(x, weight_frame.cpu_rmse, marker="o", linewidth=1.4, color="#4c78a8", label="CPU RMSE")
    axes[0].plot(x, weight_frame.memory_rmse, marker="s", linestyle="--", linewidth=1.3, color="#f58518", label="Memory RMSE")
    axes[0].set_xlabel("CPU weight")
    axes[0].set_ylabel("Resource RMSE")
    axes[0].set_xticks(x)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, weight_frame.wrmse, marker="D", linewidth=1.4, color="#b22222", label="Task-weighted WRMSE")
    axes[1].plot(x, weight_frame.balanced_wrmse, marker="^", linestyle=":", linewidth=1.3, color="#666666", label="Balanced-reference WRMSE")
    axes[1].set_xlabel("CPU weight")
    axes[1].set_ylabel("WRMSE")
    axes[1].set_xticks(x)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    export(fig, "resource_weight_sensitivity")
    plt.close(fig)

    colors = {"Random": "#8c8c8c", "Periodic": "#4c78a8", "Send-on-delta": "#f58518", "Lyapunov": "#54a24b", "MPC-DP": "#b22222"}
    markers = {"Random": "o", "Periodic": "s", "Send-on-delta": "^", "Lyapunov": "D", "MPC-DP": "X"}
    linestyles = {"Random": ":", "Periodic": "--", "Send-on-delta": "-.", "Lyapunov": (0, (5, 2)), "MPC-DP": "-"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for method in METHODS:
        data = loss_frame[loss_frame.method == method].sort_values("loss_probability")
        loss_percent = data.loss_probability * 100
        axes[0].plot(loss_percent, data.wrmse_mean, marker=markers[method], linestyle=linestyles[method], linewidth=1.3, markersize=4, color=colors[method], label=method)
        axes[0].fill_between(loss_percent, data.wrmse_mean - data.wrmse_std, data.wrmse_mean + data.wrmse_std, color=colors[method], alpha=0.10)
        axes[1].plot(loss_percent, data.mean_aoi_mean, marker=markers[method], linestyle=linestyles[method], linewidth=1.3, markersize=4, color=colors[method], label=method)
        axes[1].fill_between(loss_percent, data.mean_aoi_mean - data.mean_aoi_std, data.mean_aoi_mean + data.mean_aoi_std, color=colors[method], alpha=0.10)
    axes[0].set_xlabel("Update-message loss rate (%)")
    axes[0].set_ylabel("WRMSE")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Update-message loss rate (%)")
    axes[1].set_ylabel("Mean AoI (slots)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    export(fig, "update_loss_robustness")
    plt.close(fig)

    with (output / "figure_contract_and_qa.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n- Weight sensitivity: CPU weight varies over 0.2/0.5/0.8 while eta, B, H, data, and FITS predictions remain fixed; balanced-reference WRMSE enables schedule comparison under one common metric.\n"
            "- Loss robustness: attempted-message budget remains fixed at 189; successful updates, effective rate, WRMSE, and AoI are reported over 30 independent loss seeds.\n"
            "- Failure semantics: a lost update consumes communication budget, leaves the remote cache unchanged, and is known after ACK timeout before the next decision.\n"
        )


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}" if std > 0 else f"{mean:.{digits}f}"


def detect_hardware() -> dict[str, object]:
    hardware: dict[str, object] = {}
    if sys.platform == "win32":
        command = (
            "$c=Get-CimInstance Win32_Processor|Select-Object -First 1;"
            "$s=Get-CimInstance Win32_ComputerSystem;"
            "[pscustomobject]@{cpu=$c.Name;cores=$c.NumberOfCores;"
            "logical_processors=$c.NumberOfLogicalProcessors;"
            "memory_gb=[math]::Round($s.TotalPhysicalMemory/1GB,2)}|ConvertTo-Json -Compress"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
            )
            hardware.update(json.loads(completed.stdout.strip()))
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass
    return hardware


def write_report(
    output: Path,
    main: pd.DataFrame,
    pareto: pd.DataFrame,
    selected: pd.DataFrame,
    eta_frame: pd.DataFrame,
    horizon_frame: pd.DataFrame,
    prediction_ablation: pd.DataFrame,
    weight_sensitivity: pd.DataFrame,
    loss_summary: pd.DataFrame,
    metadata: dict[str, object],
    trace_length: int,
    validation_length: int,
    test_length: int,
) -> None:
    ordered = main.set_index("method").loc[list(METHODS)].reset_index()
    best = ordered.loc[ordered.wrmse_mean.idxmin()]
    proposed = ordered[ordered.method == "MPC-DP"].iloc[0]
    lines = [
        "# MPC-DP 状态更新策略统一实验结果（本机 py310）",
        "",
        "## 1. 实验协议",
        "",
        f"- 原始轨迹长度：{trace_length} 个时隙；验证集 {validation_length}，测试集 {test_length}。",
        "- 资源：CPU 与内存，权重均为 0.5。",
        "- 主通信预算：B=0.5；所有方法在测试段使用完全相同的消息数。",
        "- Send-on-delta 的阈值、Lyapunov 的 V0 和 MPC-DP 的 eta 仅在验证段选择。",
        "- WRMSE 使用 `sqrt(mean_t(sum_j(w_j*(x_tj-z_tj)^2)))`，没有重复加权。",
        "- MPC-DP 按论文式 (13) 将 eta 加在 `b>0` 的不发送分支，并从固定窗口预算 K 回溯；DP 状态显式保留最后更新位置。",
        "- 时延仅指更新策略自身的单时隙决策时间，不包含 FITS 推理、CSV 读取或网络传输。",
        "- Random 报告 30 个种子均值与标准差；其余方法为确定性单次运行。",
        f"- 本机环境：{metadata.get('cpu', 'unknown CPU')}，{metadata.get('cores', '?')} 核/{metadata.get('logical_processors', '?')} 线程，{metadata.get('memory_gb', '?')} GB 内存，纯 CPU。",
        "",
        "## 2. B=0.5 主结果",
        "",
        "| 方法 | WRMSE | CPU RMSE | Memory RMSE | 更新数/率 | 平均AoI | P95 AoI | 决策中位数 (ms) | 决策P95 (ms) | 预算违反 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            "| {method} | {wrmse} | {cpu} | {mem} | {count:.0f}/{rate:.4f} | {aoi} | {p95aoi} | {lat} | {p95lat} | {viol:.0f} |".format(
                method=row.method,
                wrmse=fmt_pm(row.wrmse_mean, row.wrmse_std),
                cpu=fmt_pm(row.cpu_rmse_mean, row.cpu_rmse_std),
                mem=fmt_pm(row.memory_rmse_mean, row.memory_rmse_std),
                count=row.update_count_mean,
                rate=row.update_rate_mean,
                aoi=fmt_pm(row.mean_aoi_mean, row.mean_aoi_std, 3),
                p95aoi=fmt_pm(row.p95_aoi_mean, row.p95_aoi_std, 3),
                lat=fmt_pm(row.decision_median_ms_mean, row.decision_median_ms_std, 5),
                p95lat=fmt_pm(row.decision_p95_ms_mean, row.decision_p95_ms_std, 5),
                viol=row.budget_violation_count_mean,
            )
        )
    lines.extend(
        [
            "",
            "## 3. 不同预算下的最低 WRMSE",
            "",
            "| B | 最优方法 | 最低 WRMSE | MPC-DP WRMSE |",
            "|---:|---|---:|---:|",
        ]
    )
    for budget, group in pareto.groupby("budget"):
        winner = group.loc[group.wrmse_mean.idxmin()]
        mpc = group[group.method == "MPC-DP"].iloc[0]
        lines.append(f"| {budget:.1f} | {winner.method} | {winner.wrmse_mean:.4f} | {mpc.wrmse_mean:.4f} |")
    lines.extend(
        [
            "",
            "MPC-DP 在 B=0.2、0.5、0.7 时最低；Send-on-delta 在 B=0.1、0.3 时最低。因此结果支持“中高预算下预测辅助规划具有优势”，不支持“所有预算下全面最优”。",
            "",
            "## 4. 验证集选定参数",
            "",
            "| B | Send-on-delta δ | Lyapunov V0 | MPC-DP eta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in selected.sort_values("budget").iterrows():
        lines.append(f"| {row.budget:.1f} | {row.delta:.6g} | {row.lyapunov_v0:.6g} | {row.eta:.6g} |")
    lines.extend(
        [
            "",
            "## 5. 预测输入消融（固定 B=0.5、189次更新）",
            "",
            "| 预测输入 | 验证集选择eta | 验证WRMSE | 未来预测WRMSE | 测试缓存WRMSE | CPU RMSE | Memory RMSE | 更新数/率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in prediction_ablation.iterrows():
        forecast_text = "—" if pd.isna(row.forecast_wrmse) else f"{row.forecast_wrmse:.4f}"
        lines.append(
            f"| {row.variant} | {row.selected_eta:g} | {row.validation_wrmse:.4f} | {forecast_text} | {row.wrmse:.4f} | {row.cpu_rmse:.4f} | {row.memory_rmse:.4f} | {int(row.update_count)}/{row.update_rate:.4f} |"
        )
    fits_ablation = prediction_ablation[prediction_ablation.variant == "FITS forecast"].iloc[0]
    oracle_ablation = prediction_ablation[prediction_ablation.variant == "Perfect-forecast MPC"].iloc[0]
    last_ablation = prediction_ablation[prediction_ablation.variant == "Last-value forecast"].iloc[0]
    current_ablation = prediction_ablation[prediction_ablation.variant == "Current-only (H=1)"].iloc[0]
    fits_vs_last = (last_ablation.wrmse - fits_ablation.wrmse) / last_ablation.wrmse * 100
    fits_vs_current = (current_ablation.wrmse - fits_ablation.wrmse) / current_ablation.wrmse * 100
    oracle_gap = (fits_ablation.wrmse - oracle_ablation.wrmse) / oracle_ablation.wrmse * 100
    lines.extend(
        [
            "",
            f"按相同验证集调参协议，FITS-MPC相对Last-value-MPC和Current-only的测试缓存WRMSE分别降低 {fits_vs_last:.2f}% 和 {fits_vs_current:.2f}%；相对Perfect-forecast参考仍高 {oracle_gap:.2f}%。Perfect-forecast使用真实未来信息，只是不可部署的理想参考，并非实际方法。",
            "",
            "补充的固定eta=1结果保存在 `prediction_ablation_fixed_eta.csv`。固定同一eta时Last-value可能优于FITS，说明eta与预测分布存在耦合；因此主消融采用与所有其他方法一致的验证集调参规则，而不是使用测试集选择参数。",
            "",
            "## 6. 资源权重灵敏度（固定B=0.5、eta=1、H=6）",
            "",
            "| CPU权重 | Memory权重 | 任务加权WRMSE | 0.5/0.5参考WRMSE | CPU RMSE | Memory RMSE | 平均AoI | 更新数 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in weight_sensitivity.iterrows():
        lines.append(
            f"| {row.cpu_weight:.1f} | {row.memory_weight:.1f} | {row.wrmse:.4f} | {row.balanced_wrmse:.4f} | {row.cpu_rmse:.4f} | {row.memory_rmse:.4f} | {row.mean_aoi:.3f} | {int(row.update_count)} |"
        )
    balanced_span = weight_sensitivity.balanced_wrmse.max() - weight_sensitivity.balanced_wrmse.min()
    lines.extend(
        [
            "",
            f"不同任务权重会改变MPC-DP选择的更新时隙。为避免用不同权重定义的WRMSE直接排名，额外使用统一0.5/0.5参考指标；其最大变化范围为 {balanced_span:.4f}。",
            "",
            "## 7. 更新消息丢失鲁棒性（30个随机种子）",
            "",
            "| 丢失率 | MPC-DP WRMSE | 成功更新数 | 有效更新率 | 平均AoI | P95 AoI |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    mpc_loss = loss_summary[loss_summary.method == "MPC-DP"].sort_values("loss_probability")
    for _, row in mpc_loss.iterrows():
        lines.append(
            f"| {row.loss_probability*100:.0f}% | {fmt_pm(row.wrmse_mean, row.wrmse_std)} | {fmt_pm(row.successful_update_count_mean, row.successful_update_count_std, 2)} | {row.effective_update_rate_mean:.4f} | {fmt_pm(row.mean_aoi_mean, row.mean_aoi_std, 3)} | {fmt_pm(row.p95_aoi_mean, row.p95_aoi_std, 3)} |"
        )
    mpc_zero = mpc_loss[np.isclose(mpc_loss.loss_probability, 0.0)].iloc[0]
    mpc_twenty = mpc_loss[np.isclose(mpc_loss.loss_probability, 0.2)].iloc[0]
    loss_increase = (mpc_twenty.wrmse_mean - mpc_zero.wrmse_mean) / mpc_zero.wrmse_mean * 100
    at_twenty = loss_summary[np.isclose(loss_summary.loss_probability, 0.2)]
    loss_winner = at_twenty.loc[at_twenty.wrmse_mean.idxmin()]
    lines.extend(
        [
            "",
            f"在20%消息丢失率下，MPC-DP的WRMSE相对无丢失增加 {loss_increase:.2f}%；该丢失率下五种方法中最低WRMSE为 {loss_winner.method}（{loss_winner.wrmse_mean:.4f}）。发送失败仍消耗通信预算，远端缓存保持旧值，下一时隙在ACK超时后继续决策。",
        ]
    )
    reductions = {}
    for method in ("Random", "Periodic", "Send-on-delta", "Lyapunov"):
        baseline = ordered[ordered.method == method].iloc[0].wrmse_mean
        reductions[method] = (baseline - proposed.wrmse_mean) / baseline * 100
    eta_best = eta_frame.loc[eta_frame.wrmse.idxmin()]
    horizon_best = horizon_frame.loc[horizon_frame.wrmse.idxmin()]
    main_eta = float(selected[np.isclose(selected.budget, 0.5)].iloc[0].eta)
    lines.extend(
        [
            "",
            "## 8. 结果解读",
            "",
            f"在 B=0.5 下，最低 WRMSE 的方法是 **{best.method}**（{best.wrmse_mean:.4f}）。MPC-DP 相对 Random、Periodic、Send-on-delta 和 Lyapunov 分别降低 {reductions['Random']:.2f}%、{reductions['Periodic']:.2f}%、{reductions['Send-on-delta']:.2f}% 和 {reductions['Lyapunov']:.2f}%。",
            "",
            f"主表的 eta={main_eta:g} 来自验证集；测试段灵敏度中最低点是 eta={eta_best.parameter:g}（WRMSE={eta_best.wrmse:.4f}），该点只能用于说明敏感性，不能回填主表。窗口灵敏度中 H={int(horizon_best.horizon)} 最低（WRMSE={horizon_best.wrmse:.4f}）；当前预测文件最多只有 6 步，不能伪造 H=10 结果。",
            "",
            f"MPC-DP 的决策中位数/P95 为 {proposed.decision_median_ms_mean:.4f}/{proposed.decision_p95_ms_mean:.4f} ms，但 Send-on-delta 更便宜。MPC-DP 的平均 AoI 为 {proposed.mean_aoi_mean:.3f}，高于 Periodic；预测规划把有限消息集中到变化更重要的时隙，所以误差更低但状态年龄未必最低。",
            "",
            "旧 Fig. 6-8 的数值不能与本结果拼接：旧代码存在重复加权、窗口预算重置、Lyapunov 队列可为负以及人为增加 0.5 ms 等问题，应使用本次统一结果整体替换。",
            "",
            "本机时延只能说明代码在当前 x86 CPU 上的计算成本，不能替代 ARM/工业边缘硬件部署数据。",
            "",
            "## 9. 输出文件",
            "",
            "- `main_B0.5_summary.csv`：主预算汇总。",
            "- `pareto_summary.csv`：五个预算下的 WRMSE—更新率结果。",
            "- `selected_hyperparameters.csv`：验证集选择的参数。",
            "- `eta_sensitivity.csv`、`horizon_sensitivity.csv`：灵敏度结果。",
            "- `prediction_ablation.csv`：按验证集分别选择eta的预测输入消融。",
            "- `prediction_ablation_fixed_eta.csv`：固定eta=1的控制结果。",
            "- `weight_sensitivity.csv`：CPU/内存权重灵敏度。",
            "- `update_loss_raw.csv`、`update_loss_summary.csv`：消息丢失鲁棒性原始和汇总结果。",
            "- `raw_runs.csv`：全部原始运行记录。",
            "- PNG/PDF：论文候选图。",
            "",
        ]
    )
    (output / "MPC-DP更新策略统一实验结果与分析.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "test")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "local_py310")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--random-repeats", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="run the calibrated B=0.5 main comparison only (for repeated hardware timing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual, predictions = load_inputs(args.data_dir)
    split = max(args.horizon + 1, int(round(len(actual) * args.validation_ratio)))
    if split >= len(actual) - 10:
        raise ValueError("Validation segment leaves too little test data")
    val_actual = actual[1:split]
    val_predictions = predictions[1:split]
    val_initial = actual[0]
    test_actual = actual[split:]
    test_predictions = predictions[split:]
    test_initial = actual[split - 1]

    budgets = (0.5,) if args.main_only else (0.1, 0.2, 0.3, 0.5, 0.7)
    eta_grid = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    v0_grid = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
    deltas = delta_candidates(val_actual)
    selected_rows = []
    calibration_rows = []

    for budget in budgets:
        delta, delta_frame = select_parameter(
            "Send-on-delta", deltas, val_actual, val_predictions, val_initial, budget, args.horizon
        )
        v0, v0_frame = select_parameter(
            "Lyapunov", v0_grid, val_actual, val_predictions, val_initial, budget, args.horizon
        )
        eta, eta_frame = select_parameter(
            "MPC-DP", eta_grid, val_actual, val_predictions, val_initial, budget, args.horizon
        )
        selected_rows.append({"budget": budget, "delta": delta, "lyapunov_v0": v0, "eta": eta})
        for label, frame in (("delta", delta_frame), ("v0", v0_frame), ("eta", eta_frame)):
            temp = frame.copy()
            temp["calibration_type"] = label
            calibration_rows.append(temp)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(args.output_dir / "selected_hyperparameters.csv", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_csv(args.output_dir / "calibration_details.csv", index=False)

    raw_rows = []
    for budget in budgets:
        params = selected[selected.budget == budget].iloc[0]
        repeats_for_random = 1 if args.main_only else args.random_repeats
        for seed in range(40, 40 + repeats_for_random):
            raw_rows.append(asdict(run_strategy("Random", test_actual, test_predictions, test_initial, budget, seed=seed, horizon=args.horizon)))
        raw_rows.append(asdict(run_strategy("Periodic", test_actual, test_predictions, test_initial, budget, horizon=args.horizon)))
        raw_rows.append(asdict(run_strategy("Send-on-delta", test_actual, test_predictions, test_initial, budget, parameter=params.delta, horizon=args.horizon)))
        raw_rows.append(asdict(run_strategy("Lyapunov", test_actual, test_predictions, test_initial, budget, parameter=params.lyapunov_v0, horizon=args.horizon)))
        raw_rows.append(asdict(run_strategy("MPC-DP", test_actual, test_predictions, test_initial, budget, parameter=params.eta, horizon=args.horizon)))
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(args.output_dir / "raw_runs.csv", index=False)
    pareto = aggregate_runs(raw)
    pareto.to_csv(args.output_dir / "pareto_summary.csv", index=False)
    main_summary = pareto[np.isclose(pareto.budget, 0.5)].copy()
    main_summary.to_csv(args.output_dir / "main_B0.5_summary.csv", index=False)

    if args.main_only:
        metadata = {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "data_dir": str(args.data_dir.resolve()),
            "trace_length": len(actual),
            "validation_length": len(val_actual),
            "test_length": len(test_actual),
            "weights": WEIGHTS.tolist(),
            "random_repeats": repeats_for_random,
            "loss_repeats": 0,
            "prediction_horizon": int(predictions.shape[1]),
            "execution_scope": "calibrated_B0.5_main_comparison_only",
        }
        metadata.update(detect_hardware())
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            main_summary[
                [
                    "method",
                    "wrmse_mean",
                    "cpu_rmse_mean",
                    "memory_rmse_mean",
                    "update_count_mean",
                    "update_rate_mean",
                    "mean_aoi_mean",
                    "decision_median_ms_mean",
                    "decision_p95_ms_mean",
                ]
            ].to_string(index=False)
        )
        print(f"\nMain-only results written to: {args.output_dir.resolve()}")
        return

    selected_main = selected[np.isclose(selected.budget, 0.5)].iloc[0]
    eta_rows = []
    for eta in eta_grid:
        eta_rows.append(asdict(run_strategy("MPC-DP", test_actual, test_predictions, test_initial, 0.5, parameter=eta, horizon=args.horizon)))
    eta_frame = pd.DataFrame(eta_rows)
    eta_frame.to_csv(args.output_dir / "eta_sensitivity.csv", index=False)

    horizon_rows = []
    for horizon in (1, 3, 5, min(6, predictions.shape[1])):
        horizon_rows.append(asdict(run_strategy("MPC-DP", test_actual, test_predictions, test_initial, 0.5, parameter=selected_main.eta, horizon=horizon)))
    horizon_frame = pd.DataFrame(horizon_rows).drop_duplicates(subset=["horizon"])
    horizon_frame.to_csv(args.output_dir / "horizon_sensitivity.csv", index=False)

    prediction_ablation_fixed = run_prediction_ablation(
        test_actual,
        test_predictions,
        test_initial,
        budget=0.5,
        eta=float(selected_main.eta),
        horizon=args.horizon,
    )
    prediction_ablation_fixed.to_csv(args.output_dir / "prediction_ablation_fixed_eta.csv", index=False)
    prediction_ablation = run_calibrated_prediction_ablation(
        val_actual,
        val_predictions,
        val_initial,
        test_actual,
        test_predictions,
        test_initial,
        budget=0.5,
        eta_candidates=eta_grid,
        horizon=args.horizon,
    )
    prediction_ablation.to_csv(args.output_dir / "prediction_ablation.csv", index=False)

    weight_sensitivity = run_weight_sensitivity(
        test_actual,
        test_predictions,
        test_initial,
        budget=0.5,
        eta=float(selected_main.eta),
        horizon=args.horizon,
    )
    weight_sensitivity.to_csv(args.output_dir / "weight_sensitivity.csv", index=False)

    loss_raw, loss_summary = run_loss_robustness(
        test_actual,
        test_predictions,
        test_initial,
        budget=0.5,
        delta=float(selected_main.delta),
        lyapunov_v0=float(selected_main.lyapunov_v0),
        eta=float(selected_main.eta),
        horizon=args.horizon,
        repeats=args.random_repeats,
    )
    loss_raw.to_csv(args.output_dir / "update_loss_raw.csv", index=False)
    loss_summary.to_csv(args.output_dir / "update_loss_summary.csv", index=False)

    save_plots(pareto, main_summary, eta_frame, prediction_ablation, args.output_dir)
    save_robustness_plots(weight_sensitivity, loss_summary, args.output_dir)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "data_dir": str(args.data_dir.resolve()),
        "trace_length": len(actual),
        "validation_length": len(val_actual),
        "test_length": len(test_actual),
        "weights": WEIGHTS.tolist(),
        "random_repeats": args.random_repeats,
        "loss_repeats": args.random_repeats,
        "prediction_horizon": int(predictions.shape[1]),
    }
    metadata.update(detect_hardware())
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        args.output_dir,
        main_summary,
        pareto,
        selected,
        eta_frame,
        horizon_frame,
        prediction_ablation,
        weight_sensitivity,
        loss_summary,
        metadata,
        len(actual),
        len(val_actual),
        len(test_actual),
    )
    print(main_summary[["method", "wrmse_mean", "cpu_rmse_mean", "memory_rmse_mean", "update_count_mean", "update_rate_mean", "mean_aoi_mean", "decision_median_ms_mean", "decision_p95_ms_mean"]].to_string(index=False))
    print(f"\nResults written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
