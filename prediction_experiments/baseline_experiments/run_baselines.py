from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    header: int | None
    columns: tuple[int, int] = (0, 1)


DATASETS = {
    # The first row is "2,3", i.e. the retained source-column labels written by pandas.
    # This matches the current FITS main experiment, which reads the file with header inference.
    "Alibaba": DatasetSpec("Alibaba", ROOT / "FITSTest" / "usage_2018.csv", 0),
    "GWA-T-12": DatasetSpec("GWA-T-12", ROOT / "FITSTest" / "data1.csv", 0),
    "Google": DatasetSpec("Google", ROOT / "FITSTest" / "data2.csv", None),
}


@dataclass
class ExperimentConfig:
    nrows: int = 100_000
    seq_len: int = 100
    pred_len: int = 5
    moving_average_window: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 256
    hidden_size: int = 32
    num_layers: int = 1
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 7
    min_delta: float = 1e-6
    seeds: tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)
    arima_p_candidates: tuple[int, ...] = (1, 2, 5, 10)
    arima_d_candidates: tuple[int, ...] = (0, 1)
    inference_repeats: int = 1000
    fits_cut_freq: int = 24
    fits_batch_size: int = 32
    fits_epochs: int = 20


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def causal_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data.astype(np.float32, copy=True)
    frame = pd.DataFrame(data)
    return frame.rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float32)


def load_dataset(spec: DatasetSpec, cfg: ExperimentConfig) -> tuple[np.ndarray, dict]:
    frame = pd.read_csv(spec.path, header=spec.header, nrows=cfg.nrows, usecols=list(spec.columns))
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_before = int(frame.isna().sum().sum())
    frame = frame.interpolate(method="linear", limit_direction="both").dropna()
    raw = frame.to_numpy(dtype=np.float32)
    processed = causal_moving_average(raw, cfg.moving_average_window)
    digest = hashlib.sha256(processed.tobytes()).hexdigest()
    meta = {
        "name": spec.name,
        "path": str(spec.path),
        "header": spec.header,
        "requested_nrows": cfg.nrows,
        "loaded_rows": int(len(processed)),
        "columns": [str(c) for c in frame.columns],
        "missing_values_before_interpolation": missing_before,
        "processed_sha256": digest,
        "raw_min": raw.min(axis=0).astype(float).tolist(),
        "raw_max": raw.max(axis=0).astype(float).tolist(),
        "processed_mean": processed.mean(axis=0).astype(float).tolist(),
        "processed_std": processed.std(axis=0).astype(float).tolist(),
    }
    return processed, meta


def split_series(data: np.ndarray, cfg: ExperimentConfig) -> dict[str, np.ndarray]:
    train_end = int(len(data) * cfg.train_ratio)
    val_end = train_end + int(len(data) * cfg.val_ratio)
    splits = {"train": data[:train_end], "val": data[train_end:val_end], "test": data[val_end:]}
    required = cfg.seq_len + cfg.pred_len
    for name, values in splits.items():
        if len(values) < required:
            raise ValueError(f"{name} split has {len(values)} rows; at least {required} are required")
    return splits


def make_windows(data: np.ndarray, seq_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    total = seq_len + pred_len
    windows = np.lib.stride_tricks.sliding_window_view(data, total, axis=0)
    # NumPy returns [sample, channel, time] when sliding over axis 0.
    windows = np.moveaxis(windows, -1, 1)
    x = np.ascontiguousarray(windows[:, :seq_len, :], dtype=np.float32)
    y = np.ascontiguousarray(windows[:, seq_len:, :], dtype=np.float32)
    return x, y


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_flat = y_true.reshape(-1, y_true.shape[-1]).astype(np.float64)
    pred_flat = y_pred.reshape(-1, y_pred.shape[-1]).astype(np.float64)
    error = pred_flat - true_flat
    epsilon = 1e-8
    denom = np.abs(true_flat) + np.abs(pred_flat) + epsilon
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(r2_score(true_flat, pred_flat, multioutput="uniform_average")),
        "MAPE_pct": float(np.mean(np.abs(error) / (np.abs(true_flat) + epsilon)) * 100.0),
        "sMAPE_pct": float(np.mean(2.0 * np.abs(error) / denom) * 100.0),
    }


def detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []
    global_row = {"scope": "global", "channel": "all", "horizon": "all"}
    global_row.update(metrics(y_true, y_pred))
    rows.append(global_row)
    for c in range(y_true.shape[-1]):
        row = {"scope": "channel", "channel": c, "horizon": "all"}
        row.update(metrics(y_true[..., c : c + 1], y_pred[..., c : c + 1]))
        rows.append(row)
    for h in range(y_true.shape[1]):
        row = {"scope": "horizon", "channel": "all", "horizon": h + 1}
        row.update(metrics(y_true[:, h : h + 1, :], y_pred[:, h : h + 1, :]))
        rows.append(row)
        for c in range(y_true.shape[-1]):
            row = {"scope": "channel_horizon", "channel": c, "horizon": h + 1}
            row.update(metrics(y_true[:, h : h + 1, c : c + 1], y_pred[:, h : h + 1, c : c + 1]))
            rows.append(row)
    return rows


def percentile_timing(callable_, repeats: int, warmup: int = 100) -> dict[str, float]:
    for _ in range(warmup):
        callable_()
    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        start = time.perf_counter_ns()
        callable_()
        samples[i] = (time.perf_counter_ns() - start) / 1e6
    return {
        "latency_median_ms": float(np.median(samples)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
    }


def naive_predict(x: np.ndarray, pred_len: int) -> np.ndarray:
    return np.repeat(x[:, -1:, :], pred_len, axis=1)


def fit_arima_p_d_0(series: np.ndarray, p: int, d: int) -> tuple[float, np.ndarray]:
    z = np.asarray(series, dtype=np.float64)
    if d == 1:
        z = np.diff(z)
    if len(z) <= p:
        raise ValueError("Series is too short for ARIMA order")
    target = z[p:]
    lagged = np.column_stack([z[p - lag : len(z) - lag] for lag in range(1, p + 1)])
    design = np.column_stack([np.ones(len(target)), lagged])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(coef[0]), coef[1:].astype(np.float64)


def predict_arima_p_d_0(x: np.ndarray, orders: list[tuple[int, int]], coefficients: list[tuple[float, np.ndarray]], pred_len: int) -> np.ndarray:
    n, _, channels = x.shape
    out = np.empty((n, pred_len, channels), dtype=np.float64)
    for c in range(channels):
        p, d = orders[c]
        intercept, phi = coefficients[c]
        levels = x[:, :, c].astype(np.float64)
        if d == 0:
            state = levels.copy()
            for h in range(pred_len):
                lag_values = np.stack([state[:, -lag] for lag in range(1, p + 1)], axis=1)
                forecast = intercept + lag_values @ phi
                out[:, h, c] = forecast
                state = np.concatenate([state, forecast[:, None]], axis=1)
        else:
            diff_state = np.diff(levels, axis=1)
            current_level = levels[:, -1].copy()
            for h in range(pred_len):
                lag_values = np.stack([diff_state[:, -lag] for lag in range(1, p + 1)], axis=1)
                diff_forecast = intercept + lag_values @ phi
                current_level = current_level + diff_forecast
                out[:, h, c] = current_level
                diff_state = np.concatenate([diff_state, diff_forecast[:, None]], axis=1)
    return out.astype(np.float32)


def select_arima_orders(train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, cfg: ExperimentConfig) -> tuple[list[tuple[int, int]], list[tuple[float, np.ndarray]], list[dict]]:
    orders = []
    coefficients = []
    search_rows = []
    for c in range(train.shape[1]):
        best = None
        for d in cfg.arima_d_candidates:
            for p in cfg.arima_p_candidates:
                coef = fit_arima_p_d_0(train[:, c], p, d)
                candidate_orders = [(p, d)]
                pred = predict_arima_p_d_0(
                    x_val[:, :, c : c + 1], candidate_orders, [coef], cfg.pred_len
                )
                score = metrics(y_val[:, :, c : c + 1], pred)["RMSE"]
                search_rows.append({"channel": c, "p": p, "d": d, "q": 0, "val_RMSE": score})
                if best is None or score < best[0]:
                    best = (score, (p, d), coef)
        assert best is not None
        orders.append(best[1])
        coefficients.append(best[2])
    return orders, coefficients, search_rows


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class SingleLayerGRU(nn.Module):
    def __init__(self, channels: int, hidden_size: int, pred_len: int):
        super().__init__()
        self.channels = channels
        self.pred_len = pred_len
        self.gru = nn.GRU(channels, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, pred_len * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1]).reshape(-1, self.pred_len, self.channels)


class FITSModel(nn.Module):
    """The FITS architecture used by FITSTest/duobu_1.py."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        cut_freq: int = 24,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.dominance_freq = cut_freq
        self.individual = individual
        self.length_ratio = (seq_len + pred_len) / seq_len
        output_freq = int(cut_freq * self.length_ratio)
        if individual:
            self.freq_upsampler = nn.ModuleList(
                [nn.Linear(cut_freq, output_freq, dtype=torch.cfloat) for _ in range(channels)]
            )
        else:
            self.freq_upsampler = nn.Linear(
                cut_freq,
                output_freq,
                dtype=torch.cfloat,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x, dim=1, keepdim=True)
        centered = x - x_mean
        x_var = torch.var(centered, dim=1, keepdim=True) + 1e-5
        normalized = centered / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(normalized, dim=1)
        low_specx[:, self.dominance_freq :] = 0
        low_specx = low_specx[:, : self.dominance_freq, :]
        if self.individual:
            output_freq = int(self.dominance_freq * self.length_ratio)
            low_specxy_small = torch.zeros(
                [low_specx.size(0), output_freq, self.channels],
                dtype=torch.cfloat,
                device=x.device,
            )
            for channel in range(self.channels):
                low_specxy_small[:, :, channel] = self.freq_upsampler[channel](
                    low_specx[:, :, channel]
                )
        else:
            low_specxy_small = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        target_length = self.seq_len + self.pred_len
        low_specxy = torch.zeros(
            [low_specxy_small.size(0), (target_length // 2) + 1, self.channels],
            dtype=torch.cfloat,
            device=x.device,
        )
        low_specxy[:, : low_specxy_small.size(1), :] = low_specxy_small
        low_xy = torch.fft.irfft(low_specxy, n=target_length, dim=1) * self.length_ratio
        xy = low_xy * torch.sqrt(x_var) + x_mean
        return xy[:, -self.pred_len :, :]


def train_fits(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Train FITS with its original optimization settings on the unified data protocol."""
    set_seed(seed)
    x_train, y_train = make_windows(train_data, cfg.seq_len, cfg.pred_len)
    x_val, y_val = make_windows(val_data, cfg.seq_len, cfg.pred_len)
    x_test, y_test = make_windows(test_data, cfg.seq_len, cfg.pred_len)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        WindowDataset(x_train, y_train),
        batch_size=cfg.fits_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    val_loader = DataLoader(
        WindowDataset(x_val, y_val), batch_size=cfg.fits_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        WindowDataset(x_test, y_test), batch_size=cfg.fits_batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FITSModel(
        cfg.seq_len, cfg.pred_len, train_data.shape[1], cfg.fits_cut_freq
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    history = []
    train_start = time.perf_counter()
    for epoch in range(1, cfg.fits_epochs + 1):
        model.train()
        train_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * len(xb)
        train_loss = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_sum += float(criterion(model(xb), yb).item()) * len(xb)
        val_loss = val_sum / len(val_loader.dataset)
        history.append({"epoch": epoch, "train_MSE": train_loss, "val_MSE": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        print(
            f"FITS seed={seed} epoch={epoch:02d}/{cfg.fits_epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f} best={best_loss:.6f}",
            flush=True,
        )
    train_seconds = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("FITS training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {"model_state": best_state, "seed": seed, "config": asdict(cfg)},
        output_dir / f"fits_seed_{seed}.pt",
    )
    model.to(device).eval()
    pred_batches = []
    true_batches = []
    with torch.inference_mode():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            pred_batches.append(model(xb).cpu().numpy())
            true_batches.append(yb.numpy())
    y_pred = np.concatenate(pred_batches)
    y_true = np.concatenate(true_batches)

    sample_gpu = torch.from_numpy(x_test[:1]).to(device)
    sample_cpu = torch.from_numpy(x_test[:1])
    cpu_model = copy.deepcopy(model).cpu().eval()

    def gpu_call():
        with torch.inference_mode():
            model(sample_gpu)
        if device.type == "cuda":
            torch.cuda.synchronize()

    def cpu_call():
        with torch.inference_mode():
            cpu_model(sample_cpu)

    timing = {
        "training_seconds": train_seconds,
        "epochs_run": cfg.fits_epochs,
        "best_val_MSE": best_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    timing.update({f"device_{k}": v for k, v in percentile_timing(gpu_call, cfg.inference_repeats).items()})
    timing.update({f"cpu_{k}": v for k, v in percentile_timing(cpu_call, cfg.inference_repeats).items()})
    return y_true.astype(np.float32), y_pred.astype(np.float32), history, timing


def train_gru(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    set_seed(seed)
    mean = train_data.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_data.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0

    def scale(values: np.ndarray) -> np.ndarray:
        return ((values - mean) / std).astype(np.float32)

    x_train, y_train = make_windows(scale(train_data), cfg.seq_len, cfg.pred_len)
    x_val, y_val = make_windows(scale(val_data), cfg.seq_len, cfg.pred_len)
    x_test, y_test = make_windows(scale(test_data), cfg.seq_len, cfg.pred_len)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        WindowDataset(x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    val_loader = DataLoader(WindowDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(WindowDataset(x_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleLayerGRU(train_data.shape[1], cfg.hidden_size, cfg.pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    train_start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * len(xb)
        train_loss = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_sum += float(criterion(model(xb), yb).item()) * len(xb)
        val_loss = val_sum / len(val_loader.dataset)
        history.append({"epoch": epoch, "train_MSE_scaled": train_loss, "val_MSE_scaled": val_loss})
        if val_loss < best_loss - cfg.min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"GRU seed={seed} epoch={epoch:02d} train={train_loss:.6f} val={val_loss:.6f} "
            f"best={best_loss:.6f} patience={bad_epochs}/{cfg.patience}",
            flush=True,
        )
        if bad_epochs >= cfg.patience:
            break
    train_seconds = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("GRU training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {"model_state": best_state, "mean": mean, "std": std, "seed": seed, "config": asdict(cfg)},
        output_dir / f"gru_seed_{seed}.pt",
    )
    model.to(device).eval()
    pred_batches = []
    true_batches = []
    with torch.inference_mode():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            pred_batches.append(model(xb).cpu().numpy())
            true_batches.append(yb.numpy())
    pred_scaled = np.concatenate(pred_batches)
    true_scaled = np.concatenate(true_batches)
    y_pred = pred_scaled * std[None, None, :] + mean[None, None, :]
    y_true = true_scaled * std[None, None, :] + mean[None, None, :]

    sample_gpu = torch.from_numpy(x_test[:1]).to(device)
    sample_cpu = torch.from_numpy(x_test[:1])
    cpu_model = copy.deepcopy(model).cpu().eval()

    def gpu_call():
        with torch.inference_mode():
            model(sample_gpu)
        if device.type == "cuda":
            torch.cuda.synchronize()

    def cpu_call():
        with torch.inference_mode():
            cpu_model(sample_cpu)

    timing = {
        "training_seconds": train_seconds,
        "epochs_run": len(history),
        "best_val_MSE_scaled": best_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    timing.update({f"device_{k}": v for k, v in percentile_timing(gpu_call, cfg.inference_repeats).items()})
    timing.update({f"cpu_{k}": v for k, v in percentile_timing(cpu_call, cfg.inference_repeats).items()})
    return y_true.astype(np.float32), y_pred.astype(np.float32), history, timing


def environment_info() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def save_prediction(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    np.savez_compressed(path, y_true=y_true, y_pred=y_pred)


def run_dataset(spec: DatasetSpec, cfg: ExperimentConfig, models: set[str], out_root: Path) -> tuple[list[dict], list[dict], dict]:
    print(f"\n===== DATASET {spec.name} =====", flush=True)
    dataset_dir = out_root / spec.name.replace("/", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data, meta = load_dataset(spec, cfg)
    splits = split_series(data, cfg)
    x_val, y_val = make_windows(splits["val"], cfg.seq_len, cfg.pred_len)
    x_test, y_test = make_windows(splits["test"], cfg.seq_len, cfg.pred_len)
    meta["split_rows"] = {name: int(len(values)) for name, values in splits.items()}
    meta["window_counts"] = {
        "train": int(len(splits["train"]) - cfg.seq_len - cfg.pred_len + 1),
        "val": int(len(x_val)),
        "test": int(len(x_test)),
    }
    (dataset_dir / "dataset_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = []
    detailed_rows = []

    if "naive" in models:
        start = time.perf_counter()
        pred = naive_predict(x_test, cfg.pred_len)
        total_seconds = time.perf_counter() - start
        result = metrics(y_test, pred)
        sample = x_test[:1]
        timing = percentile_timing(lambda: naive_predict(sample, cfg.pred_len), cfg.inference_repeats)
        row = {
            "dataset": spec.name,
            "model": "NaiveLastValue",
            "seed": "deterministic",
            **result,
            **timing,
            "test_total_seconds": total_seconds,
            "parameter_count": 0,
        }
        summary_rows.append(row)
        for detail in detailed_metrics(y_test, pred):
            detailed_rows.append({"dataset": spec.name, "model": "NaiveLastValue", "seed": "deterministic", **detail})
        save_prediction(dataset_dir / "naive_predictions.npz", y_test, pred)
        print(f"Naive: {result}", flush=True)

    if "arima" in models:
        search_start = time.perf_counter()
        orders, coefficients, search_rows = select_arima_orders(splits["train"], x_val, y_val, cfg)
        selection_seconds = time.perf_counter() - search_start
        pd.DataFrame(search_rows).to_csv(dataset_dir / "arima_validation_search.csv", index=False)
        start = time.perf_counter()
        pred = predict_arima_p_d_0(x_test, orders, coefficients, cfg.pred_len)
        total_seconds = time.perf_counter() - start
        result = metrics(y_test, pred)
        sample = x_test[:1]
        timing = percentile_timing(
            lambda: predict_arima_p_d_0(sample, orders, coefficients, cfg.pred_len), cfg.inference_repeats
        )
        parameter_count = int(sum(1 + len(phi) for _, phi in coefficients))
        row = {
            "dataset": spec.name,
            "model": "ARIMA",
            "seed": "deterministic",
            **result,
            **timing,
            "test_total_seconds": total_seconds,
            "selection_seconds": selection_seconds,
            "parameter_count": parameter_count,
            "selected_orders": json.dumps([(p, d, 0) for p, d in orders]),
        }
        summary_rows.append(row)
        for detail in detailed_metrics(y_test, pred):
            detailed_rows.append({"dataset": spec.name, "model": "ARIMA", "seed": "deterministic", **detail})
        save_prediction(dataset_dir / "arima_predictions.npz", y_test, pred)
        (dataset_dir / "arima_model.json").write_text(
            json.dumps(
                {
                    "estimator": "conditional least squares",
                    "orders": [(p, d, 0) for p, d in orders],
                    "coefficients": [
                        {"intercept": intercept, "phi": phi.tolist()} for intercept, phi in coefficients
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"ARIMA orders={[(p, d, 0) for p, d in orders]}: {result}", flush=True)

    if "fits" in models:
        for seed in cfg.seeds:
            y_true, pred, history, timing = train_fits(
                splits["train"], splits["val"], splits["test"], cfg, seed, dataset_dir
            )
            pd.DataFrame(history).to_csv(dataset_dir / f"fits_seed_{seed}_history.csv", index=False)
            result = metrics(y_true, pred)
            row = {"dataset": spec.name, "model": "FITS", "seed": seed, **result, **timing}
            summary_rows.append(row)
            for detail in detailed_metrics(y_true, pred):
                detailed_rows.append({"dataset": spec.name, "model": "FITS", "seed": seed, **detail})
            save_prediction(dataset_dir / f"fits_seed_{seed}_predictions.npz", y_true, pred)
            print(f"FITS seed={seed}: {result}", flush=True)

    if "gru" in models:
        for seed in cfg.seeds:
            y_true, pred, history, timing = train_gru(
                splits["train"], splits["val"], splits["test"], cfg, seed, dataset_dir
            )
            pd.DataFrame(history).to_csv(dataset_dir / f"gru_seed_{seed}_history.csv", index=False)
            result = metrics(y_true, pred)
            row = {"dataset": spec.name, "model": "SingleLayerGRU", "seed": seed, **result, **timing}
            summary_rows.append(row)
            for detail in detailed_metrics(y_true, pred):
                detailed_rows.append({"dataset": spec.name, "model": "SingleLayerGRU", "seed": seed, **detail})
            save_prediction(dataset_dir / f"gru_seed_{seed}_predictions.npz", y_true, pred)
            print(f"GRU seed={seed}: {result}", flush=True)

    return summary_rows, detailed_rows, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified FITS, Naive, ARIMA and single-layer GRU experiments")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=["fits", "naive", "arima", "gru"], default=["fits", "naive", "arima", "gru"])
    parser.add_argument("--nrows", type=int, default=100_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--inference-repeats", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "unified_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        nrows=args.nrows,
        seeds=tuple(args.seeds),
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        inference_repeats=args.inference_repeats,
    )
    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "environment.json").write_text(
        json.dumps(environment_info(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_summary = []
    all_detailed = []
    metadata = {}
    start = time.perf_counter()
    for dataset_name in args.datasets:
        summary, detailed, meta = run_dataset(DATASETS[dataset_name], cfg, set(args.models), out_root)
        all_summary.extend(summary)
        all_detailed.extend(detailed)
        metadata[dataset_name] = meta
        pd.DataFrame(all_summary).to_csv(out_root / "metrics_raw.csv", index=False)
        pd.DataFrame(all_detailed).to_csv(out_root / "metrics_detailed_raw.csv", index=False)

    raw = pd.DataFrame(all_summary)
    numeric_metrics = ["RMSE", "MAE", "R2", "MAPE_pct", "sMAPE_pct"]
    summary_records = []
    for (dataset, model), group in raw.groupby(["dataset", "model"], sort=False):
        record = {"dataset": dataset, "model": model, "runs": len(group)}
        for metric_name in numeric_metrics:
            record[f"{metric_name}_mean"] = float(group[metric_name].mean())
            record[f"{metric_name}_std"] = float(group[metric_name].std(ddof=1)) if len(group) > 1 else 0.0
        for metric_name in ["latency_median_ms", "latency_p95_ms", "cpu_latency_median_ms", "cpu_latency_p95_ms", "parameter_count"]:
            if metric_name in group and group[metric_name].notna().any():
                record[f"{metric_name}_mean"] = float(group[metric_name].dropna().mean())
        summary_records.append(record)
    pd.DataFrame(summary_records).to_csv(out_root / "metrics_summary.csv", index=False)
    run_info = {
        "elapsed_seconds": time.perf_counter() - start,
        "datasets": args.datasets,
        "models": args.models,
        "status": "complete",
    }
    (out_root / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCOMPLETE: {out_root}", flush=True)


if __name__ == "__main__":
    main()
