from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
DATASETS = {
    "Alibaba": "alibaba",
    "GWA-T-12": "gwa_t_12",
    "Google": "google",
}
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
MODEL_LABELS = {
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def arima_predict(x, model_spec, pred_len: int):
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


def verify_predictions(unified, benchmark_dir: Path, package_dir: Path) -> dict:
    checks = {}
    for dataset, slug in DATASETS.items():
        inputs = np.load(package_dir / slug / "heldout_input_windows.npz")
        x_raw = np.asarray(inputs["x_raw"], dtype=np.float32)
        indices = np.asarray(inputs["source_window_index"], dtype=np.int64)
        dataset_dir = benchmark_dir / dataset
        checks[dataset] = {}
        for key in MODEL_KEYS:
            if key == "naive":
                actual = np.repeat(x_raw[:, -1:, :], int(inputs["pred_len"]), axis=1)
                reference_path = dataset_dir / "naive_predictions.npz"
            elif key == "arima":
                model_spec = json.loads((dataset_dir / "arima_model.json").read_text())
                actual = arima_predict(x_raw, model_spec, int(inputs["pred_len"]))
                reference_path = dataset_dir / "arima_predictions.npz"
            else:
                checkpoint_path = dataset_dir / f"{key}_seed_2026.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                cfg = unified.UnifiedConfig(**checkpoint["config"])
                mean = np.asarray(checkpoint["mean"], dtype=np.float32)
                std = np.asarray(checkpoint["std"], dtype=np.float32)
                model = unified.make_model(key, cfg, channels=x_raw.shape[-1]).cpu().eval()
                model.load_state_dict(checkpoint["model_state"])
                batches = []
                with torch.inference_mode():
                    scaled = ((x_raw - mean) / std).astype(np.float32)
                    for start in range(0, len(scaled), 128):
                        batches.append(model(torch.from_numpy(scaled[start : start + 128])).numpy())
                actual = np.concatenate(batches) * std[None, None, :] + mean[None, None, :]
                reference_path = dataset_dir / f"{key}_seed_2026_predictions.npz"
            reference = np.load(reference_path)["y_pred"][indices]
            error = np.abs(np.asarray(actual, dtype=np.float32) - reference)
            # Saved predictions were produced by CUDA kernels, while this audit
            # replays the frozen checkpoints on CPU. Recurrent kernels may
            # accumulate backend-dependent rounding differences, so use a
            # scale-normalized replay error rather than bitwise equality.
            replay_rmse = float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))
            reference_range = float(np.ptp(reference))
            normalized_rmse = replay_rmse / max(reference_range, 1e-6)
            passed = bool(np.isfinite(actual).all() and normalized_rmse <= 1e-3)
            checks[dataset][key] = {
                "windows": int(len(indices)),
                "values": int(error.size),
                "max_absolute_error": float(error.max()),
                "mean_absolute_error": float(error.mean()),
                "replay_rmse": replay_rmse,
                "reference_range": reference_range,
                "normalized_rmse": normalized_rmse,
                "acceptance_threshold_normalized_rmse": 1e-3,
                "passed": passed,
            }
            if not passed:
                raise RuntimeError(
                    f"Prediction conformance failed: {dataset}/{key}; "
                    f"max={float(error.max())}, mean={float(error.mean())}, "
                    f"normalized_rmse={normalized_rmse}"
                )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=HERE / "results" / "industrial_three_datasets_i5_1235u_baremetal_cpu_formal_v1",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=HERE / "results" / "industrial_three_dataset_package_v1",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=HERE / "results" / "all_models_unified_full_v1",
    )
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    package_dir = args.package_dir.resolve()
    benchmark_dir = args.benchmark_dir.resolve()
    unified = load_module("industrial_validation_unified", HERE / "run_all_models_unified.py")

    raw_frames = []
    summary_frames = []
    environments = {}
    for dataset, slug in DATASETS.items():
        raw = pd.read_csv(result_dir / slug / "vm_inference_raw.csv")
        summary = pd.read_csv(result_dir / slug / "vm_inference_summary.csv")
        environment = json.loads(
            (result_dir / slug / "vm_inference_environment.json").read_text(encoding="utf-8")
        )
        if len(raw) != 45 or len(summary) != 9:
            raise RuntimeError(f"Incomplete rows for {dataset}: raw={len(raw)}, summary={len(summary)}")
        if set(raw["model_key"]) != set(MODEL_KEYS) or not (raw.groupby("model_key").size() == 5).all():
            raise RuntimeError(f"Incomplete model/run cells for {dataset}")
        if raw.duplicated(["model_key", "run"]).any():
            raise RuntimeError(f"Duplicate model/run cells for {dataset}")
        if not raw["output_finite"].astype(bool).all():
            raise RuntimeError(f"Non-finite output recorded for {dataset}")
        if environment["dataset"] != dataset or environment["input_windows"] != 257:
            raise RuntimeError(f"Environment/input metadata mismatch for {dataset}")
        raw["dataset"] = dataset
        summary.insert(0, "dataset", dataset)
        raw_frames.append(raw)
        summary_frames.append(summary)
        environments[dataset] = environment

    raw_all = pd.concat(raw_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    if len(raw_all) != 135 or raw_all.duplicated(["dataset", "model_key", "run"]).any():
        raise RuntimeError("The combined 135-row protocol is incomplete or duplicated")

    manifest = json.loads((package_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    system_before = (result_dir / "system_before.txt").read_text(encoding="utf-8")
    missing_hashes = []
    for item in manifest["datasets"].values():
        if item["input_sha256"] not in system_before:
            missing_hashes.append(item["input_sha256"])
        for checkpoint in item["checkpoints"].values():
            if checkpoint["sha256"] not in system_before:
                missing_hashes.append(checkpoint["sha256"])
    if missing_hashes:
        raise RuntimeError(f"Remote hash log is missing {len(missing_hashes)} expected artifacts")

    prediction_checks = verify_predictions(unified, benchmark_dir, package_dir)
    raw_all.to_csv(result_dir / "industrial_three_dataset_raw_135.csv", index=False)
    summary_all.to_csv(result_dir / "industrial_three_dataset_summary_27.csv", index=False)
    validation = {
        "status": "passed",
        "datasets": 3,
        "models": 9,
        "independent_process_runs_per_model_dataset": 5,
        "raw_rows": int(len(raw_all)),
        "summary_rows": int(len(summary_all)),
        "input_windows_per_dataset": 257,
        "timed_inferences_per_process": 2000,
        "total_timed_inferences": int(len(raw_all) * 2000),
        "all_outputs_finite": True,
        "no_duplicate_cells": True,
        "remote_artifact_hashes_match": True,
        "prediction_conformance": prediction_checks,
        "environments": environments,
        "files": {
            "raw": "industrial_three_dataset_raw_135.csv",
            "summary": "industrial_three_dataset_summary_27.csv",
        },
    }
    (result_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": validation["status"],
        "raw_rows": validation["raw_rows"],
        "summary_rows": validation["summary_rows"],
        "total_timed_inferences": validation["total_timed_inferences"],
        "result_dir": str(result_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
