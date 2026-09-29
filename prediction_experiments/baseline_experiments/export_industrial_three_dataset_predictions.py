from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("industrial_export_unified", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def neural_predict(unified, key: str, checkpoint_path: Path, x_raw: np.ndarray, batch_size: int):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_key"] != key or int(checkpoint["seed"]) != 2026:
        raise RuntimeError(f"Unexpected checkpoint identity: {checkpoint_path}")
    cfg = unified.UnifiedConfig(**checkpoint["config"])
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    model = unified.make_model(key, cfg, channels=x_raw.shape[-1]).cpu().eval()
    model.load_state_dict(checkpoint["model_state"])
    scaled = ((x_raw - mean) / std).astype(np.float32)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(scaled), batch_size):
            outputs.append(model(torch.from_numpy(scaled[start : start + batch_size])).numpy())
    prediction_scaled = np.concatenate(outputs)
    return (prediction_scaled * std[None, None, :] + mean[None, None, :]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    unified_path = base_dir / "run_all_models_unified.py"
    unified = load_module(unified_path)
    manifest = {
        "status": "complete",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "threads": args.threads,
        "batch_size_for_export_only": args.batch_size,
        "note": "Export batch size does not alter the earlier batch=1 latency benchmark.",
        "source_hashes": {
            "run_all_models_unified.py": sha256_file(unified_path),
            "run_baselines.py": sha256_file(base_dir / "run_baselines.py"),
            "export_industrial_three_dataset_predictions.py": sha256_file(Path(__file__).resolve()),
        },
        "datasets": {},
    }

    for dataset, slug in DATASETS.items():
        dataset_dir = base_dir / slug
        checkpoint_dir = dataset_dir / "checkpoints"
        input_path = dataset_dir / "heldout_input_windows.npz"
        payload = np.load(input_path)
        x_raw = np.asarray(payload["x_raw"], dtype=np.float32)
        indices = np.asarray(payload["source_window_index"], dtype=np.int64)
        pred_len = int(payload["pred_len"])
        if x_raw.shape != (257, 100, 2) or not np.isfinite(x_raw).all():
            raise RuntimeError(f"Unexpected input for {dataset}: {x_raw.shape}")
        dataset_output = output_dir / slug
        dataset_output.mkdir()
        dataset_manifest = {
            "dataset": dataset,
            "input_file_sha256": sha256_file(input_path),
            "input_array_sha256": sha256_array(x_raw),
            "source_window_index_sha256": sha256_array(indices),
            "input_shape": list(x_raw.shape),
            "models": {},
        }
        for key in MODEL_KEYS:
            checkpoint_path = None
            if key == "naive":
                prediction = np.repeat(x_raw[:, -1:, :], pred_len, axis=1).astype(np.float32)
            elif key == "arima":
                checkpoint_path = checkpoint_dir / "arima_model.json"
                model_spec = json.loads(checkpoint_path.read_text())
                prediction = arima_predict(x_raw, model_spec, pred_len)
            else:
                checkpoint_path = checkpoint_dir / f"{key}_seed_2026.pt"
                prediction = neural_predict(unified, key, checkpoint_path, x_raw, args.batch_size)
            if prediction.shape != (257, pred_len, 2) or not np.isfinite(prediction).all():
                raise RuntimeError(f"Invalid prediction from {dataset}/{key}: {prediction.shape}")
            prediction_path = dataset_output / f"{key}_predictions.npz"
            np.savez_compressed(
                prediction_path,
                dataset=np.asarray(dataset),
                model_key=np.asarray(key),
                source_window_index=indices,
                y_pred=prediction,
            )
            dataset_manifest["models"][key] = {
                "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
                "prediction_shape": list(prediction.shape),
                "prediction_dtype": str(prediction.dtype),
                "prediction_array_sha256": sha256_array(prediction),
                "prediction_file_sha256": sha256_file(prediction_path),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
                "output_all_finite": True,
            }
            print(f"EXPORTED {dataset}/{key} {prediction_path}", flush=True)
        manifest["datasets"][dataset] = dataset_manifest

    manifest_path = output_dir / "remote_prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
