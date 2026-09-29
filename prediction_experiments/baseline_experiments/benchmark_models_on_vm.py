from __future__ import annotations

"""Inference-only benchmark for the unified nine-model checkpoints.

The controller launches one fresh worker process for every model/run.  This
avoids one model inheriting allocations or caches from another model.  The
script is intended for the 2-vCPU/2-GB Ubuntu VM and never trains a model.
"""

import argparse
import csv
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


MODEL_KEYS = (
    "fits",
    "fits_delta",
    "naive",
    "arima",
    "gru",
    "dlinear",
    "s2s_lstm",
    "frets",
    "patchtst",
)

LABELS = {
    "fits": "FITS",
    "fits_delta": "FITS-Delta",
    "naive": "Naive last-value",
    "arima": "ARIMA",
    "gru": "Single-layer GRU",
    "dlinear": "DLinear",
    "s2s_lstm": "S2S LSTM",
    "frets": "FreTS-lite",
    "patchtst": "PatchTST-lite",
}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def arima_predict(x, model_spec, pred_len: int):
    import numpy as np

    orders = [(int(p), int(d)) for p, d, _ in model_spec["orders"]]
    coefficients = [
        (float(item["intercept"]), np.asarray(item["phi"], dtype=np.float64))
        for item in model_spec["coefficients"]
    ]
    n, _, channels = x.shape
    out = np.empty((n, pred_len, channels), dtype=np.float64)
    for channel in range(channels):
        p, d = orders[channel]
        intercept, phi = coefficients[channel]
        levels = x[:, :, channel].astype(np.float64)
        if d == 0:
            state = levels.copy()
            for horizon in range(pred_len):
                lags = np.stack([state[:, -lag] for lag in range(1, p + 1)], axis=1)
                forecast = intercept + lags @ phi
                out[:, horizon, channel] = forecast
                state = np.concatenate([state, forecast[:, None]], axis=1)
        else:
            state = np.diff(levels, axis=1)
            level = levels[:, -1].copy()
            for horizon in range(pred_len):
                lags = np.stack([state[:, -lag] for lag in range(1, p + 1)], axis=1)
                forecast = intercept + lags @ phi
                level = level + forecast
                out[:, horizon, channel] = level
                state = np.concatenate([state, forecast[:, None]], axis=1)
    return out.astype(np.float32)


def worker(args: argparse.Namespace) -> None:
    import numpy as np

    npz = np.load(args.input_npz)
    x_raw = np.asarray(npz["x_raw"], dtype=np.float32)
    if x_raw.ndim == 2:
        x_raw = x_raw[None, ...]
    if x_raw.ndim != 3 or x_raw.shape[0] < 1:
        raise ValueError(f"x_raw must have shape [windows, time, channels], got {x_raw.shape}")
    if not np.isfinite(x_raw).all():
        raise ValueError("x_raw contains NaN or infinite values")
    pred_len = int(npz["pred_len"])
    input_windows = int(x_raw.shape[0])
    key = args.worker_model
    parameter_count = 0
    state_bytes = 0

    if key == "naive":
        def call(index: int):
            sample_raw = x_raw[index : index + 1]
            return np.repeat(sample_raw[:, -1:, :], pred_len, axis=1)

    elif key == "arima":
        model_spec = json.loads((args.checkpoint_dir / "arima_model.json").read_text())
        parameter_count = sum(1 + len(item["phi"]) for item in model_spec["coefficients"])
        state_bytes = parameter_count * 8

        def call(index: int):
            return arima_predict(x_raw[index : index + 1], model_spec, pred_len)

    else:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        import importlib.util
        import torch

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        module_path = Path(__file__).resolve().parent / "run_all_models_unified.py"
        spec = importlib.util.spec_from_file_location("unified_models", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        checkpoint_path = args.checkpoint_dir / f"{key}_seed_{args.seed}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = module.UnifiedConfig(**checkpoint["config"])
        model = module.make_model(key, cfg, channels=x_raw.shape[-1]).cpu().eval()
        model.load_state_dict(checkpoint["model_state"])
        parameter_count = module.count_parameters(model)
        state_bytes = module.model_state_bytes(model)
        mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        std = np.asarray(checkpoint["std"], dtype=np.float32)
        sample_pool = torch.from_numpy(((x_raw - mean) / std).astype(np.float32))

        def call(index: int):
            with torch.inference_mode():
                return model(sample_pool[index : index + 1])

    for iteration in range(args.warmup):
        call(iteration % input_windows)
    validation_result = call(0)
    if hasattr(validation_result, "detach"):
        validation_array = validation_result.detach().cpu().numpy()
    else:
        validation_array = np.asarray(validation_result)
    if not np.isfinite(validation_array).all():
        raise RuntimeError(f"{key} produced NaN or infinite predictions")
    timings = []
    for iteration in range(args.repeats):
        start = time.perf_counter_ns()
        result = call(iteration % input_windows)
        if hasattr(result, "shape"):
            _ = result.shape
        timings.append((time.perf_counter_ns() - start) / 1e6)
    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "model_key": key,
        "model": LABELS[key],
        "run": args.worker_run,
        "seed": args.seed if key not in {"naive", "arima"} else "deterministic",
        "latency_median_ms": statistics.median(timings),
        "latency_mean_ms": statistics.fmean(timings),
        "latency_p95_ms": percentile(timings, 0.95),
        "latency_min_ms": min(timings),
        "latency_max_ms": max(timings),
        "process_peak_rss_mb": max_rss_kib / 1024.0,
        "parameter_count": parameter_count,
        "model_state_bytes": state_bytes,
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dataset": args.dataset,
        "input_windows": input_windows,
        "output_finite": True,
    }
    print(json.dumps(payload, ensure_ascii=False))


def controller(args: argparse.Namespace) -> None:
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
    })
    for run in range(1, args.runs + 1):
        for key in args.models:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-model", key,
                "--worker-run", str(run),
                "--checkpoint-dir", str(args.checkpoint_dir),
                "--input-npz", str(args.input_npz),
                "--seed", str(args.seed),
                "--threads", str(args.threads),
                "--warmup", str(args.warmup),
                "--repeats", str(args.repeats),
                "--dataset", args.dataset,
            ]
            completed = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            rows.append(payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    raw_path = args.output / "vm_inference_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for key in args.models:
        group = [row for row in rows if row["model_key"] == key]
        summary.append({
            "model": LABELS[key],
            "runs": len(group),
            "parameter_count": group[0]["parameter_count"],
            "model_state_kib": group[0]["model_state_bytes"] / 1024.0,
            "latency_median_ms_mean": statistics.fmean(row["latency_median_ms"] for row in group),
            "latency_median_ms_std": statistics.stdev(row["latency_median_ms"] for row in group) if len(group) > 1 else 0.0,
            "latency_p95_ms_mean": statistics.fmean(row["latency_p95_ms"] for row in group),
            "process_peak_rss_mb_mean": statistics.fmean(row["process_peak_rss_mb"] for row in group),
            "process_peak_rss_mb_max": max(row["process_peak_rss_mb"] for row in group),
        })
    summary_path = args.output / "vm_inference_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    metadata = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count_visible": os.cpu_count(),
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "independent_process_runs": args.runs,
        "batch_size": 1,
        "dataset": args.dataset,
        "input_windows": int(np.load(args.input_npz)["x_raw"].shape[0]),
        "input_selection": "evenly spaced held-out test windows, cycled one at a time",
        "input_length": 100,
        "prediction_length": 5,
        "channels": 2,
        "checkpoint_dataset": args.dataset,
        "checkpoint_seed": args.seed,
        "note": "Peak RSS includes the isolated Python inference process and its runtime libraries.",
    }
    (args.output / "vm_inference_environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"COMPLETE {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/vm_inference_all_models"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--dataset", default="unspecified")
    parser.add_argument("--worker-model", choices=MODEL_KEYS)
    parser.add_argument("--worker-run", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_model:
        worker(parsed)
    else:
        controller(parsed)
