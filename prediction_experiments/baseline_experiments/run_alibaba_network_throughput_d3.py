from __future__ import annotations

"""Proof-of-concept three-resource prediction and state-update experiment.

The main manuscript experiments remain CPU/memory (d=2).  This entry point
uses Alibaba machine_usage.csv to add a node-level network-throughput state and
checks the dimensional extensibility of FITS-Delta and the update policies.
It deliberately does not claim to measure end-to-end available bandwidth.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
RAFT_ROOT = WORKSPACE / "raft"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(RAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(RAFT_ROOT))

UNIFIED_SPEC = importlib.util.spec_from_file_location(
    "network_d3_unified_models", HERE / "run_all_models_unified.py"
)
UNIFIED = importlib.util.module_from_spec(UNIFIED_SPEC)
assert UNIFIED_SPEC.loader is not None
sys.modules[UNIFIED_SPEC.name] = UNIFIED
UNIFIED_SPEC.loader.exec_module(UNIFIED)

from edgebench_core.update_policies import (  # noqa: E402
    ExactTraceBudgetGuard,
    mpc_dp_first_action,
    normalize_weights,
    weighted_squared_error,
)


CHANNELS = ("CPU", "Memory", "Network")
WEIGHTS_D3 = np.asarray([1.0 / 3.0] * 3, dtype=float)
METHODS = ("Periodic", "Send-on-delta", "Lyapunov", "MPC-DP")
ETA_GRID = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
V0_GRID = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
OFFICIAL_ARCHIVE_SHA256 = "3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardware_info() -> dict[str, object]:
    info: dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    if sys.platform == "win32":
        command = (
            "$c=Get-CimInstance Win32_Processor|Select-Object -First 1;"
            "$s=Get-CimInstance Win32_ComputerSystem;"
            "[pscustomobject]@{cpu=$c.Name;cores=$c.NumberOfCores;"
            "logical=$c.NumberOfLogicalProcessors;"
            "memory_gb=[math]::Round($s.TotalPhysicalMemory/1GB,2)}|ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
            )
            info.update(json.loads(result.stdout.strip()))
        except Exception as exc:  # hardware metadata is non-critical
            info["hardware_query_error"] = str(exc)
    return info


def causal_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32, copy=True)
    return (
        pd.DataFrame(values)
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )


def prepare_data(
    raw_csv: Path,
    processed_dir: Path,
    machine_count: int,
    rows_per_machine: int,
    moving_average: int,
    seed: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    npz_path = processed_dir / "alibaba_network_d3.npz"
    manifest_path = processed_dir / "data_manifest.json"
    selected_csv = processed_dir / "alibaba_network_d3_selected.csv"
    if npz_path.exists() and manifest_path.exists() and not force:
        saved = np.load(npz_path, allow_pickle=False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return saved["values"], saved["timestamps"], saved["machine_ids"], manifest

    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)
    rng = np.random.default_rng(seed)
    # Keep a bounded random candidate pool in memory, but scan the complete CSV
    # before choosing machines.  Stopping after the first populated chunk would
    # bias selection toward the machine IDs stored near the beginning of the
    # archive.
    candidate_numbers = rng.choice(np.arange(1, 4001), size=100, replace=False)
    candidate_ids = [f"m_{number}" for number in candidate_numbers]
    candidate_set = set(candidate_ids)
    # One extra observation is required to construct a causal traffic rate from
    # consecutive normalized network counters; retain a duplicate buffer too.
    target_buffer = rows_per_machine + 501
    buffers: dict[str, list[pd.DataFrame]] = {machine_id: [] for machine_id in candidate_ids}
    counts = {machine_id: 0 for machine_id in candidate_ids}
    complete: set[str] = set()
    names = ["machine_id", "timestamp", "cpu", "memory", "net_in", "net_out"]

    print(f"Scanning {raw_csv} for {machine_count} seed-selected machines...", flush=True)
    reader = pd.read_csv(
        raw_csv,
        header=None,
        usecols=[0, 1, 2, 3, 6, 7],
        names=names,
        chunksize=1_000_000,
        low_memory=False,
    )
    chunks_read = 0
    for chunks_read, chunk in enumerate(reader, start=1):
        chunk = chunk.loc[chunk["machine_id"].isin(candidate_set)].copy()
        if chunk.empty:
            continue
        for column in names[1:]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna()
        valid = (
            chunk[["cpu", "memory", "net_in", "net_out"]]
            .ge(0.0)
            .all(axis=1)
            & chunk[["cpu", "memory", "net_in", "net_out"]].le(100.0).all(axis=1)
        )
        chunk = chunk[valid]
        for machine_id, group in chunk.groupby("machine_id", sort=False):
            remaining = target_buffer - counts[machine_id]
            if remaining <= 0:
                continue
            part = group.head(remaining).copy()
            if not part.empty:
                buffers[machine_id].append(part)
                counts[machine_id] += len(part)
                if counts[machine_id] >= target_buffer:
                    complete.add(machine_id)
        if chunks_read % 10 == 0:
            print(
                f"  chunks={chunks_read}, complete candidates={len(complete)}",
                flush=True,
            )
        # Deliberately scan the complete file so that selection is determined by
        # the seeded candidate order rather than the physical CSV row order.

    ordered_complete = [machine_id for machine_id in candidate_ids if machine_id in complete]
    if len(ordered_complete) < machine_count:
        raise RuntimeError(
            f"Only {len(ordered_complete)} candidate machines reached {target_buffer} valid rows"
        )
    selected_ids = ordered_complete[:machine_count]
    series = []
    timestamp_series = []
    export_frames = []
    for machine_id in selected_ids:
        frame = pd.concat(buffers[machine_id], ignore_index=True)
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if len(frame) < rows_per_machine:
            raise RuntimeError(f"{machine_id} has only {len(frame)} unique valid rows")
        frame = frame.iloc[: rows_per_machine + 1].copy()
        timestamps_raw = frame["timestamp"].to_numpy(dtype=np.float64)
        elapsed = np.diff(timestamps_raw)
        if np.any(elapsed <= 0):
            raise RuntimeError(f"{machine_id} contains non-increasing timestamps")
        net_in_raw = frame["net_in"].to_numpy(dtype=np.float64)
        net_out_raw = frame["net_out"].to_numpy(dtype=np.float64)
        # The selected raw network series are nondecreasing.  Convert consecutive
        # increments to a rate proxy so that the third state describes current
        # traffic intensity rather than the level of a cumulative counter.
        net_in_rate = np.maximum(np.diff(net_in_raw), 0.0) / elapsed
        net_out_rate = np.maximum(np.diff(net_out_raw), 0.0) / elapsed
        network_rate = net_in_rate + net_out_rate
        aligned = frame.iloc[1:].copy()
        aligned["net_in_rate"] = net_in_rate
        aligned["net_out_rate"] = net_out_rate
        aligned["network_rate_raw"] = network_rate
        cpu_memory = aligned[["cpu", "memory"]].to_numpy(dtype=np.float32) / 100.0
        series.append(np.column_stack([cpu_memory, network_rate]).astype(np.float32))
        timestamp_series.append(aligned["timestamp"].to_numpy(dtype=np.int64))
        export_frames.append(aligned)

    values = np.stack(series).astype(np.float32)
    # Fit the traffic-rate scale only on the chronological training portion.
    # A robust upper quantile avoids allowing rare long gaps or bursts to set the
    # full dynamic range.  Test values above this training scale are clipped.
    train_end = int(rows_per_machine * 0.70)
    network_scale = float(np.quantile(values[:, :train_end, 2], 0.99))
    if not np.isfinite(network_scale) or network_scale <= 0.0:
        raise RuntimeError(f"Invalid training-derived network-rate scale: {network_scale}")
    values[..., 2] = np.clip(values[..., 2] / network_scale, 0.0, 1.0)
    for machine_index, frame in enumerate(export_frames):
        frame["cpu"] = values[machine_index, :, 0]
        frame["memory"] = values[machine_index, :, 1]
        frame["network"] = values[machine_index, :, 2]
        frame["net_in"] = frame["net_in"] / 100.0
        frame["net_out"] = frame["net_out"] / 100.0
    values = np.stack(
        [causal_average(machine, moving_average) for machine in values]
    ).astype(np.float32)
    timestamps = np.stack(timestamp_series)
    machine_ids = np.asarray(selected_ids)
    np.savez_compressed(npz_path, values=values, timestamps=timestamps, machine_ids=machine_ids)
    pd.concat(export_frames, ignore_index=True).to_csv(selected_csv, index=False)
    manifest: dict[str, object] = {
        "purpose": "proof-of-concept d=3 resource-state experiment",
        "raw_csv": str(raw_csv.resolve()),
        "raw_csv_size_bytes": raw_csv.stat().st_size,
        "official_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "selection_seed": seed,
        "candidate_pool_size": len(candidate_ids),
        "chunks_read": chunks_read,
        "machine_count": machine_count,
        "rows_per_machine": rows_per_machine,
        "selected_machine_ids": selected_ids,
        "channels": list(CHANNELS),
        "network_definition": "(positive delta(net_in) + positive delta(net_out)) / elapsed seconds",
        "network_scale": network_scale,
        "network_normalization": "divide by the 99th percentile fitted on the first 70% of selected-machine observations, then clip to [0,1]",
        "normalization": "CPU and memory divided by 100; network uses a training-only robust traffic-rate scale",
        "raw_network_monotonic_fraction": float(
            np.mean(
                np.concatenate(
                    [
                        np.diff(frame["net_in"].to_numpy()) >= 0.0
                        for frame in export_frames
                    ]
                )
            )
        ),
        "moving_average": f"causal rolling mean, window={moving_average}, applied per machine",
        "invalid_rule": "drop missing values and values outside [0,100] before selection",
        "npz_sha256": sha256_file(npz_path),
        "selected_csv_sha256": sha256_file(selected_csv),
        "values_shape": list(values.shape),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return values, timestamps, machine_ids, manifest


def make_split_windows(
    values: np.ndarray,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    rows = values.shape[1]
    train_end = int(rows * train_ratio)
    val_end = train_end + int(rows * val_ratio)
    segments = {
        "train": values[:, :train_end],
        "val": values[:, train_end:val_end],
        "test": values[:, val_end:],
    }
    minimum_segment_length = seq_len + pred_len
    too_short = {
        split_name: int(segment.shape[1])
        for split_name, segment in segments.items()
        if segment.shape[1] < minimum_segment_length
    }
    if too_short:
        raise ValueError(
            "Each chronological split must contain at least "
            f"seq_len + pred_len = {minimum_segment_length} rows; got {too_short}."
        )
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, segment in segments.items():
        x_rows, y_rows = [], []
        for machine in segment:
            x, y = UNIFIED.BASE.make_windows(machine, seq_len, pred_len)
            x_rows.append(x)
            y_rows.append(y)
        result[split_name] = (np.stack(x_rows), np.stack(y_rows))
    train_points = segments["train"].reshape(-1, values.shape[-1])
    mean = train_points.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_points.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return result, mean, std


def scaled(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((array - mean) / std).astype(np.float32)


def predict_batches(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    original_shape = x.shape[:2]
    flat = x.reshape(-1, *x.shape[2:])
    loader = DataLoader(TensorDataset(torch.from_numpy(flat)), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            outputs.append(model(batch).cpu().numpy())
    prediction = np.concatenate(outputs, axis=0)
    return prediction.reshape(*original_shape, *prediction.shape[1:])


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    row: dict[str, float] = {}
    channel_rmse = []
    for index, channel in enumerate(CHANNELS):
        metrics = UNIFIED.BASE.metrics(y_true[..., index : index + 1], y_pred[..., index : index + 1])
        prefix = channel.lower()
        row[f"{prefix}_rmse"] = metrics["RMSE"]
        row[f"{prefix}_mae"] = metrics["MAE"]
        row[f"{prefix}_r2"] = metrics["R2"]
        channel_rmse.append(metrics["RMSE"])
    global_metrics = UNIFIED.BASE.metrics(y_true, y_pred)
    row["global_rmse"] = global_metrics["RMSE"]
    row["global_mae"] = global_metrics["MAE"]
    row["global_r2"] = global_metrics["R2"]
    third_prefix = CHANNELS[2].lower()
    row[f"{third_prefix}_nrmse"] = row[f"{third_prefix}_rmse"]  # channels are scaled to [0,1]
    row["macro_nrmse"] = float(np.mean(channel_rmse))
    return row


def train_or_load_fits_delta(
    seed: int,
    windows: dict[str, tuple[np.ndarray, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
    output: Path,
    seq_len: int,
    pred_len: int,
    cut_freq: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
) -> tuple[nn.Module, dict[str, object], np.ndarray, np.ndarray]:
    checkpoint_path = output / f"fits_delta_d3_seed_{seed}.pt"
    model = UNIFIED.DeltaResidualFITS(seq_len, pred_len, len(CHANNELS), cut_freq).cpu()
    train_x, train_y = windows["train"]
    val_x, val_y = windows["val"]
    test_x, _ = windows["test"]
    train_x_s = scaled(train_x, mean, std).reshape(-1, seq_len, len(CHANNELS))
    train_y_s = scaled(train_y, mean, std).reshape(-1, pred_len, len(CHANNELS))
    val_x_s = scaled(val_x, mean, std)
    test_x_s = scaled(test_x, mean, std)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        training_meta = dict(checkpoint["training_meta"])
        print(f"Reusing {checkpoint_path.name}", flush=True)
    else:
        UNIFIED.BASE.set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x_s), torch.from_numpy(train_y_s)),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        bad_epochs = 0
        started = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            model.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
            val_prediction_s = predict_batches(model, val_x_s, batch_size)
            val_loss = float(np.mean(np.square(val_prediction_s - scaled(val_y, mean, std))))
            print(
                f"FITS-Delta d3 seed={seed} epoch={epoch:02d} val_MSE={val_loss:.7f}",
                flush=True,
            )
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break
        if best_state is None:
            raise RuntimeError("FITS-Delta produced no valid checkpoint")
        model.load_state_dict(best_state)
        training_meta = {
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "best_validation_mse_scaled": best_loss,
            "training_seconds": time.perf_counter() - started,
        }
        torch.save(
            {
                "model_state": best_state,
                "mean": mean,
                "std": std,
                "channels": CHANNELS,
                "training_meta": training_meta,
                "config": {
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "cut_freq": cut_freq,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_epochs": max_epochs,
                    "patience": patience,
                },
            },
            checkpoint_path,
        )

    model.eval()
    val_prediction = predict_batches(model, val_x_s, batch_size) * std + mean
    test_prediction = predict_batches(model, test_x_s, batch_size) * std + mean
    sample = torch.from_numpy(test_x_s.reshape(-1, seq_len, len(CHANNELS))[:1])
    timing = UNIFIED.timed_inference(model, sample, repeats=1000, warmup=100)
    training_meta.update(timing)
    training_meta["parameter_count"] = UNIFIED.count_parameters(model)
    training_meta["model_state_bytes"] = UNIFIED.model_state_bytes(model)
    np.savez_compressed(
        output / f"fits_delta_d3_seed_{seed}_predictions.npz",
        val_prediction=val_prediction.astype(np.float32),
        test_prediction=test_prediction.astype(np.float32),
    )
    return model, training_meta, val_prediction, test_prediction


def run_update_strategy(
    method: str,
    actual: np.ndarray,
    forecast: np.ndarray,
    initial_cache: np.ndarray,
    budget: float,
    parameter: float | None,
) -> dict[str, float | int | str]:
    weights = normalize_weights(WEIGHTS_D3)
    guard = ExactTraceBudgetGuard(len(actual), budget)
    cache = initial_cache.astype(float, copy=True)
    queue = 0.0
    age = 0
    errors, aois, latencies = [], [], []
    for t in range(len(actual)):
        start = time.perf_counter_ns()
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
        latencies.append((time.perf_counter_ns() - start) / 1e6)
    errors_a = np.asarray(errors)
    if guard.used != guard.target or guard.violations:
        raise AssertionError(f"Budget failure for {method}")
    per_channel = np.sqrt(np.mean(np.square(errors_a), axis=0))
    third_resource_metric = f"{CHANNELS[2].lower()}_rmse"
    return {
        "method": method,
        "wrmse_d3": float(np.sqrt(np.mean(np.sum(weights * np.square(errors_a), axis=1)))),
        "cpu_rmse": float(per_channel[0]),
        "memory_rmse": float(per_channel[1]),
        third_resource_metric: float(per_channel[2]),
        "update_count": guard.used,
        "update_rate": guard.used / len(actual),
        "mean_aoi": float(np.mean(aois)),
        "p95_aoi": float(np.percentile(aois, 95)),
        "decision_median_ms": float(np.median(latencies)),
        "decision_p95_ms": float(np.percentile(latencies, 95)),
        "resource_payload_bytes_float32": len(CHANNELS) * 4,
    }


def candidate_deltas(val_current: np.ndarray) -> np.ndarray:
    differences = np.diff(val_current, axis=1)
    norms = np.sqrt(np.sum(WEIGHTS_D3 * np.square(differences), axis=2)).reshape(-1)
    return np.unique(np.concatenate(([0.0], np.quantile(norms, np.linspace(0, 1, 31)))))


def calibrate(
    method: str,
    candidates: tuple[float, ...] | np.ndarray,
    actual: np.ndarray,
    forecasts: np.ndarray,
    budget: float,
) -> tuple[float, list[dict[str, float | str]]]:
    rows: list[dict[str, float | str]] = []
    for candidate in candidates:
        outcomes = [
            run_update_strategy(
                method,
                actual[index],
                forecasts[index],
                actual[index, 0],
                budget,
                float(candidate),
            )
            for index in range(len(actual))
        ]
        rows.append(
            {
                "method": method,
                "parameter": float(candidate),
                "validation_wrmse_mean": float(np.mean([row["wrmse_d3"] for row in outcomes])),
                "validation_aoi_mean": float(np.mean([row["mean_aoi"] for row in outcomes])),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["validation_wrmse_mean"]),
            float(row["validation_aoi_mean"]),
            float(row["parameter"]),
        ),
    )
    return float(ranked[0]["parameter"]), rows


def mean_std_ci(values: pd.Series) -> tuple[float, float, float, float]:
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, std, mean - half, mean + half


def summarize_update(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    third_resource_metric = f"{CHANNELS[2].lower()}_rmse"
    metrics = [
        "wrmse_d3",
        "cpu_rmse",
        "memory_rmse",
        third_resource_metric,
        "update_count",
        "update_rate",
        "mean_aoi",
        "p95_aoi",
        "decision_median_ms",
        "decision_p95_ms",
        "resource_payload_bytes_float32",
    ]
    machine_level = raw.groupby(["method", "machine_id"], as_index=False)[metrics].mean()
    rows = []
    for method, group in machine_level.groupby("method", sort=False):
        row: dict[str, float | str | int] = {
            "method": method,
            "independent_machines": len(group),
        }
        for metric in metrics:
            mean, std, low, high = mean_std_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return machine_level, pd.DataFrame(rows)


def paired_wilcoxon_holm(
    machine_level: pd.DataFrame,
    reference: str = "MPC-DP",
) -> pd.DataFrame:
    """Compare machine-level WRMSE with paired two-sided Wilcoxon tests."""

    pivot = machine_level.pivot(index="machine_id", columns="method", values="wrmse_d3")
    rows: list[dict[str, float | str | int]] = []
    for baseline in [method for method in METHODS if method != reference]:
        difference = pivot[reference] - pivot[baseline]
        statistic, p_value = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
        rows.append(
            {
                "reference": reference,
                "baseline": baseline,
                "independent_machines": len(difference),
                "mean_difference_reference_minus_baseline": float(difference.mean()),
                "median_difference_reference_minus_baseline": float(difference.median()),
                "wilcoxon_statistic": float(statistic),
                "p_value_two_sided": float(p_value),
            }
        )
    order = np.argsort([float(row["p_value_two_sided"]) for row in rows])
    previous = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * float(rows[index]["p_value_two_sided"]))
        adjusted = max(previous, adjusted)
        rows[index]["p_value_holm"] = adjusted
        previous = adjusted
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    manifest: dict[str, object],
    prediction_summary: pd.DataFrame,
    update_summary: pd.DataFrame,
    selected: pd.DataFrame,
    environment: dict[str, object],
) -> None:
    prediction_order = ["FITS-Delta", "NaiveLastValue"]
    pred = prediction_summary.set_index("model").loc[prediction_order].reset_index()
    update = update_summary.set_index("method").loc[list(METHODS)].reset_index()
    lines = [
        "# Alibaba网络吞吐量第三资源维度实验结果与分析",
        "",
        "## 1. 研究目的与边界",
        "",
        "本实验检验FITS-Delta预测与状态更新机制能否从CPU/内存二维状态扩展到CPU、内存和节点级网络吞吐量三维状态。它不测量端到端可用带宽，也不构成三维候选发现或三维端到端卸载实验。",
        "",
        "## 2. 数据与协议",
        "",
        f"- 数据源：`{manifest['raw_csv']}`。",
        f"- 独立机器：{manifest['machine_count']}台；每台{manifest['rows_per_machine']}个按时间排序的有效观测。",
        f"- 机器选择种子：{manifest['selection_seed']}；机器ID：{', '.join(manifest['selected_machine_ids'])}。",
        "- 三维状态：CPU、内存和Network。由于所选原始网络序列呈非递减计数器特征，Network定义为`net_in`与`net_out`相邻正增量之和除以实际时间间隔，并用训练段99%分位数归一化。",
        "- 所有窗口均在单台机器内部构造，不跨机器边界。",
        "- 每台机器按70%/15%/15%进行时间顺序训练/验证/测试划分。",
        "- FITS-Delta：输入100步、预测5步、cut frequency=24、5个训练种子；Naive重复窗口最后观测。",
        "- 更新预算B=0.5，三维等权重[1/3,1/3,1/3]；所有方法使用完全相同的更新次数。",
        "- Send-on-delta、Lyapunov和MPC-DP参数仅在验证段选择，测试段不参与调参。",
        f"- 运行环境：{environment.get('cpu', 'unknown CPU')}，PyTorch {environment['torch']}，{environment['torch_threads']}个推理线程。",
        "",
        "## 3. 三维预测结果",
        "",
        "| 模型 | Network NRMSE↓ | Network MAE↓ | Macro NRMSE↓ | CPU RMSE↓ | Memory RMSE↓ | 参数量 | CPU中位/P95时延(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in pred.iterrows():
        lines.append(
            f"| {row.model} | {row.network_nrmse_mean:.4g} | {row.network_mae_mean:.4g} | "
            f"{row.macro_nrmse_mean:.4f} | {row.cpu_rmse_mean:.4f} | {row.memory_rmse_mean:.4f} | "
            f"{int(round(row.parameter_count_mean))} | {row.latency_median_ms_mean:.4f}/{row.latency_p95_ms_mean:.4f} |"
        )
    lines.extend(
        [
            "",
            "NRMSE以[0,1]资源范围归一化；因此数值0.01相当于约1个百分点的利用率误差。该对比用于确认第三维预测输入有效，而不是重新进行九模型选型。",
            "",
            "## 4. 三维状态更新结果",
            "",
            "| 更新策略 | 3D WRMSE↓ (95% CI) | Network RMSE↓ | 平均AoI↓ | 更新率 | 决策中位/P95(ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in update.iterrows():
        lines.append(
            f"| {row.method} | {row.wrmse_d3_mean:.4f} "
            f"({row.wrmse_d3_ci95_low:.4f}–{row.wrmse_d3_ci95_high:.4f}) | "
            f"{row.network_rmse_mean:.4g} | {row.mean_aoi_mean:.3f} | "
            f"{row.update_rate_mean:.4f} | {row.decision_median_ms_mean:.5f}/{row.decision_p95_ms_mean:.5f} |"
        )
    winner = update.loc[update.wrmse_d3_mean.idxmin()]
    lines.extend(
        [
            "",
            f"在相同B=0.5预算下，三维WRMSE点估计最低的方法为{winner.method}（{winner.wrmse_d3_mean:.4f}）。该结果只用于本三维实验内部比较，不能与原二维WRMSE直接比较。",
            "",
            "## 5. 验证集选择参数",
            "",
            "| 预测种子 | Send-on-delta阈值 | Lyapunov V0 | MPC-DP eta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in selected.sort_values("seed").iterrows():
        lines.append(
            f"| {int(row.seed)} | {row.delta:.6g} | {row.lyapunov_v0:.6g} | {row.eta:.6g} |"
        )
    lines.extend(
        [
            "",
            "## 6. 能够支持与不能支持的结论",
            "",
            "本实验能够支持：FITS-Delta和四种状态更新策略的代码能够处理显式三维资源状态；网络吞吐量预测误差、三维缓存误差和更新代价均可被独立量化。",
            "",
            "本实验不能支持：Network变量等同于端到端可用带宽；三维候选发现或完整卸载链路已经完成；增加网络维度必然改善二维主实验结果。搜索协议本身不依赖资源维数，但若在候选发现中使用Network约束，仍需扩展可行性谓词与Top-k确认载荷。",
            "",
            "## 7. 自动检查",
            "",
            f"- 数据形状：{manifest['values_shape']}。",
            f"- 独立统计单位：{int(update.independent_machines.min())}台机器。",
            f"- 预测记录：{len(prediction_summary)}个模型汇总；更新方法：{len(update_summary)}种。",
            "- 所有数值均为有限值；每种方法在每条轨迹上使用相同的精确更新预算。",
            "- float32资源向量净载荷由二维8字节增至三维12字节；该数值不包含协议头。",
            "",
        ]
    )
    (output / "Alibaba网络吞吐量第三资源维度实验结果与分析.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=HERE / "data" / "raw" / "alibaba_2018" / "machine_usage.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=HERE / "data" / "processed" / "alibaba_network_d3_rate_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "alibaba_network_throughput_d3_formal_v2_20260915",
    )
    parser.add_argument("--machine-count", type=int, default=20)
    parser.add_argument("--rows-per-machine", type=int, default=5000)
    parser.add_argument("--moving-average", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--cut-freq", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values, timestamps, machine_ids, manifest = prepare_data(
        args.raw_csv,
        args.processed_dir,
        args.machine_count,
        args.rows_per_machine,
        args.moving_average,
        args.selection_seed,
        args.force_prepare,
    )
    if values.shape != (args.machine_count, args.rows_per_machine, 3):
        raise AssertionError(values.shape)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise AssertionError("Prepared values must be finite and within [0,1]")
    if not np.all(np.diff(timestamps, axis=1) > 0):
        raise AssertionError("Timestamps must increase strictly within every machine")

    windows, mean, std = make_split_windows(values, args.seq_len, args.pred_len, 0.70, 0.15)
    np.savez_compressed(args.output_dir / "split_statistics.npz", train_mean=mean, train_std=std)
    environment = hardware_info()
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prediction_rows = []
    seed_predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    _, val_y = windows["val"]
    test_x, test_y = windows["test"]
    val_x, _ = windows["val"]
    for seed in args.seeds:
        model, meta, val_prediction, test_prediction = train_or_load_fits_delta(
            seed,
            windows,
            mean,
            std,
            args.output_dir,
            args.seq_len,
            args.pred_len,
            args.cut_freq,
            args.batch_size,
            args.max_epochs,
            args.patience,
            args.learning_rate,
        )
        row: dict[str, object] = {"model": "FITS-Delta", "seed": seed, **metric_row(test_y, test_prediction)}
        row.update(
            {
                "parameter_count": meta["parameter_count"],
                "model_state_bytes": meta["model_state_bytes"],
                "latency_median_ms": meta["latency_median_ms"],
                "latency_p95_ms": meta["latency_p95_ms"],
                "training_seconds": meta["training_seconds"],
                "best_epoch": meta["best_epoch"],
            }
        )
        prediction_rows.append(row)
        seed_predictions[seed] = (val_prediction, test_prediction)

    naive_prediction = np.repeat(test_x[:, :, -1:, :], args.pred_len, axis=2)
    started = time.perf_counter_ns()
    for _ in range(1100):
        _ = np.repeat(test_x[0, 0, -1:, :], args.pred_len, axis=0)
    naive_total_ms = (time.perf_counter_ns() - started) / 1e6 / 1100
    prediction_rows.append(
        {
            "model": "NaiveLastValue",
            "seed": "deterministic",
            **metric_row(test_y, naive_prediction),
            "parameter_count": 0,
            "model_state_bytes": 0,
            "latency_median_ms": naive_total_ms,
            "latency_p95_ms": naive_total_ms,
            "training_seconds": 0.0,
            "best_epoch": 0,
        }
    )
    prediction_raw = pd.DataFrame(prediction_rows)
    prediction_raw.to_csv(args.output_dir / "prediction_raw.csv", index=False)
    metric_columns = [column for column in prediction_raw.columns if column not in {"model", "seed"}]
    prediction_summary = (
        prediction_raw.groupby("model", sort=False)[metric_columns]
        .agg(["mean", "std"])
    )
    prediction_summary.columns = [f"{name}_{stat}" for name, stat in prediction_summary.columns]
    prediction_summary = prediction_summary.reset_index().fillna(0.0)
    prediction_summary.to_csv(args.output_dir / "prediction_summary.csv", index=False)

    val_current = val_x[:, :, -1, :]
    test_current = test_x[:, :, -1, :]
    calibration_rows = []
    selected_rows = []
    update_rows = []
    delta_grid = candidate_deltas(val_current)
    for seed in args.seeds:
        val_prediction, test_prediction = seed_predictions[seed]
        val_forecast = np.concatenate([val_current[:, :, None, :], val_prediction], axis=2)
        test_forecast = np.concatenate([test_current[:, :, None, :], test_prediction], axis=2)
        delta, delta_rows = calibrate(
            "Send-on-delta", delta_grid, val_current, val_forecast, args.budget
        )
        v0, v0_rows = calibrate("Lyapunov", V0_GRID, val_current, val_forecast, args.budget)
        eta, eta_rows = calibrate("MPC-DP", ETA_GRID, val_current, val_forecast, args.budget)
        for rows in (delta_rows, v0_rows, eta_rows):
            for row in rows:
                row["seed"] = seed
                calibration_rows.append(row)
        selected_rows.append(
            {"seed": seed, "delta": delta, "lyapunov_v0": v0, "eta": eta}
        )
        parameters = {
            "Periodic": None,
            "Send-on-delta": delta,
            "Lyapunov": v0,
            "MPC-DP": eta,
        }
        for machine_index, machine_id in enumerate(machine_ids):
            for method in METHODS:
                result = run_update_strategy(
                    method,
                    test_current[machine_index],
                    test_forecast[machine_index],
                    test_current[machine_index, 0],
                    args.budget,
                    parameters[method],
                )
                result.update({"seed": seed, "machine_id": str(machine_id)})
                update_rows.append(result)

    calibration = pd.DataFrame(calibration_rows)
    selected = pd.DataFrame(selected_rows)
    update_raw = pd.DataFrame(update_rows)
    calibration.to_csv(args.output_dir / "calibration_details.csv", index=False)
    selected.to_csv(args.output_dir / "selected_hyperparameters.csv", index=False)
    update_raw.to_csv(args.output_dir / "update_raw.csv", index=False)
    machine_level, update_summary = summarize_update(update_raw)
    machine_level.to_csv(args.output_dir / "update_machine_level.csv", index=False)
    update_summary.to_csv(args.output_dir / "update_summary.csv", index=False)
    paired_wilcoxon_holm(machine_level).to_csv(
        args.output_dir / "paired_wilcoxon_holm.csv", index=False
    )

    numeric_prediction = prediction_raw.select_dtypes(include=[np.number]).to_numpy()
    numeric_update = update_raw.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric_prediction).all() or not np.isfinite(numeric_update).all():
        raise AssertionError("Non-finite output detected")
    counts = update_raw.groupby(["method", "seed", "machine_id"]).size()
    if not (counts == 1).all() or len(update_raw) != len(METHODS) * len(args.seeds) * len(machine_ids):
        raise AssertionError("Missing or duplicate update runs")
    per_trace_updates = update_raw.groupby(["seed", "machine_id"])["update_count"].nunique()
    if not (per_trace_updates == 1).all():
        raise AssertionError("Update budgets differ across methods")

    write_report(
        args.output_dir,
        manifest,
        prediction_summary,
        update_summary,
        selected,
        environment,
    )
    run_manifest = {
        "script": str(globals().get("ENTRY_SCRIPT", Path(__file__).resolve())),
        "arguments": vars(args),
        "data_manifest": manifest,
        "environment": environment,
        "checks": {
            "finite_outputs": True,
            "strictly_increasing_timestamps": True,
            "no_cross_machine_windows": True,
            "exact_matched_update_budget": True,
            "update_run_count": len(update_raw),
            "prediction_run_count": len(prediction_raw),
        },
    }
    run_manifest["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in run_manifest["arguments"].items()}
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Completed. Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
