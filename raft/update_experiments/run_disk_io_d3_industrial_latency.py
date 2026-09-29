"""Replay the formal Alibaba disk-I/O d=3 update experiment for timing.

This script deliberately does not train or run a forecasting model.  It loads
the frozen FITS-Delta forecasts and validation-selected policy parameters from
the formal experiment, reconstructs the held-out test states, and replays the
four update policies.  The resulting non-timing metrics are used as a
cross-platform consistency check; only decision latency is hardware-specific.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgebench_core.update_policies import (  # noqa: E402
    ExactTraceBudgetGuard,
    mpc_dp_first_action,
    normalize_weights,
    weighted_squared_error,
)


METHODS = ("Periodic", "Send-on-delta", "Lyapunov", "MPC-DP")
SEEDS = (2026, 2027, 2028, 2029, 2030)
WEIGHTS = np.asarray([1.0 / 3.0] * 3, dtype=float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--pred-len", type=int, default=5)
    return parser.parse_args()


def load_selected_parameters(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            rows[seed] = {
                "delta": float(row["delta"]),
                "lyapunov_v0": float(row["lyapunov_v0"]),
                "eta": float(row["eta"]),
            }
    if set(rows) != set(SEEDS):
        raise ValueError(f"Expected parameters for {SEEDS}, got {sorted(rows)}")
    return rows


def reconstruct_test_current(values: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    if values.ndim != 3 or values.shape[0] != 20 or values.shape[2] != 3:
        raise ValueError(f"Unexpected value shape: {values.shape}")
    rows = values.shape[1]
    train_end = int(rows * 0.70)
    val_end = train_end + int(rows * 0.15)
    test = values[:, val_end:]
    window_count = test.shape[1] - seq_len - pred_len + 1
    if window_count <= 0:
        raise ValueError("Test split is too short")
    # The current state is the last element of each causal input window.
    return test[:, seq_len - 1 : seq_len - 1 + window_count].astype(float)


def run_strategy(
    method: str,
    actual: np.ndarray,
    forecast: np.ndarray,
    initial_cache: np.ndarray,
    budget: float,
    parameter: float | None,
) -> dict[str, float | int | str]:
    weights = normalize_weights(WEIGHTS)
    guard = ExactTraceBudgetGuard(len(actual), budget)
    cache = np.asarray(initial_cache, dtype=float).copy()
    queue = 0.0
    age = 0
    errors: list[np.ndarray] = []
    aois: list[int] = []
    latencies_ms: list[float] = []

    for t in range(len(actual)):
        start_ns = time.perf_counter_ns()
        if method == "Periodic":
            proposed = math.floor((t + 1) * guard.target / len(actual)) > math.floor(
                t * guard.target / len(actual)
            )
        elif method == "Send-on-delta":
            if parameter is None:
                raise ValueError("Send-on-delta requires a threshold")
            proposed = math.sqrt(weighted_squared_error(actual[t], cache, weights)) > parameter
        elif method == "Lyapunov":
            if parameter is None:
                raise ValueError("Lyapunov requires V0")
            error_if_silent = weighted_squared_error(actual[t], cache, weights)
            vt = parameter * (t + 2) ** 0.5
            proposed = queue * (1.0 - budget) < vt * error_if_silent + queue * (0.0 - budget)
        elif method == "MPC-DP":
            if parameter is None:
                raise ValueError("MPC-DP requires eta")
            local_budget = min(
                guard.target - guard.used,
                int(math.ceil(budget * len(forecast[t]))),
            )
            proposed = mpc_dp_first_action(
                forecast[t], cache, local_budget, parameter, weights
            )
        else:
            raise ValueError(method)

        action = guard.apply(bool(proposed), t)
        if method == "Lyapunov":
            queue = max(queue + float(action) - budget, 0.0)
        if action:
            cache = actual[t].copy()
            age = 0
        else:
            age += 1
        errors.append(actual[t] - cache)
        aois.append(age)
        latencies_ms.append((time.perf_counter_ns() - start_ns) / 1e6)

    if guard.used != guard.target or guard.violations:
        raise AssertionError(f"Budget failure for {method}")
    error_array = np.asarray(errors, dtype=float)
    per_channel = np.sqrt(np.mean(np.square(error_array), axis=0))
    return {
        "method": method,
        "wrmse_d3": float(
            np.sqrt(np.mean(np.sum(weights * np.square(error_array), axis=1)))
        ),
        "cpu_rmse": float(per_channel[0]),
        "memory_rmse": float(per_channel[1]),
        "diskio_rmse": float(per_channel[2]),
        "update_count": int(guard.used),
        "update_rate": float(guard.used / len(actual)),
        "mean_aoi": float(np.mean(aois)),
        "p95_aoi": float(np.percentile(aois, 95)),
        "decision_median_ms": float(np.median(latencies_ms)),
        "decision_p95_ms": float(np.percentile(latencies_ms, 95)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    value_path = args.data_dir / "alibaba_disk_io_d3.npz"
    parameter_path = args.data_dir / "selected_hyperparameters.csv"
    saved = np.load(value_path, allow_pickle=False)
    values = saved["values"]
    machine_ids = saved["machine_ids"]
    current = reconstruct_test_current(values, args.seq_len, args.pred_len)
    selected = load_selected_parameters(parameter_path)

    rows: list[dict[str, object]] = []
    input_hashes = {
        value_path.name: sha256(value_path),
        parameter_path.name: sha256(parameter_path),
    }
    for seed in SEEDS:
        prediction_path = args.data_dir / f"fits_delta_d3_seed_{seed}_predictions.npz"
        prediction = np.load(prediction_path, allow_pickle=False)["test_prediction"].astype(float)
        input_hashes[prediction_path.name] = sha256(prediction_path)
        if prediction.shape[:2] != current.shape[:2] or prediction.shape[2:] != (
            args.pred_len,
            3,
        ):
            raise ValueError(f"Prediction/current mismatch for seed {seed}: {prediction.shape}")
        forecast = np.concatenate([current[:, :, None, :], prediction], axis=2)
        parameters = {
            "Periodic": None,
            "Send-on-delta": selected[seed]["delta"],
            "Lyapunov": selected[seed]["lyapunov_v0"],
            "MPC-DP": selected[seed]["eta"],
        }
        for machine_index, machine_id in enumerate(machine_ids):
            for method in METHODS:
                result = run_strategy(
                    method,
                    current[machine_index],
                    forecast[machine_index],
                    current[machine_index, 0],
                    args.budget,
                    parameters[method],
                )
                result.update({"seed": seed, "machine_id": str(machine_id)})
                rows.append(result)

    write_csv(args.output_dir / "update_raw.csv", rows)
    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {"method": method, "run_count": len(method_rows)}
        for metric in (
            "wrmse_d3",
            "cpu_rmse",
            "memory_rmse",
            "diskio_rmse",
            "update_count",
            "update_rate",
            "mean_aoi",
            "p95_aoi",
            "decision_median_ms",
            "decision_p95_ms",
        ):
            summary[f"{metric}_mean"] = float(
                np.mean([float(row[metric]) for row in method_rows])
            )
        summary_rows.append(summary)
    write_csv(args.output_dir / "update_summary.csv", summary_rows)

    metadata = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "budget": args.budget,
        "weights": WEIGHTS.tolist(),
        "machine_count": int(len(machine_ids)),
        "test_windows_per_machine": int(current.shape[1]),
        "seeds": list(SEEDS),
        "input_sha256": input_hashes,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
