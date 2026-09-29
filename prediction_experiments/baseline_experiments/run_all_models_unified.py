from __future__ import annotations

"""Unified re-run of every prediction model used in the manuscript.

The model definitions are adapted from the author's existing scripts while the
data split, preprocessing, checkpoint selection, metric implementation and
latency protocol are shared by every model.

Source mapping
--------------
FITS                 FITSTest/duobu_1.py via run_baselines.FITSModel
FITS-Delta           baseline_experiments/run_fixed_parameter_fits.py
DLinear              FITSTest/DLinear.py (batch-isolation bug fixed)
S2S LSTM             LSTM_D.py
FreTS                 FITSTest/FreTS.py (the author's lightweight variant)
PatchTST              FITSTest/7.py (the author's current edge-sized variant)
Single-layer GRU      baseline_experiments/run_baselines.py
Naive / ARIMA         baseline_experiments/run_baselines.py
"""

import argparse
import copy
import importlib.util
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
BASE_SPEC = importlib.util.spec_from_file_location("unified_base", HERE / "run_baselines.py")
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
sys.modules[BASE_SPEC.name] = BASE
BASE_SPEC.loader.exec_module(BASE)


MODEL_LABELS = {
    "fits": "FITS",
    "fits_delta": "FITS-Delta",
    "naive": "NaiveLastValue",
    "arima": "ARIMA",
    "gru": "SingleLayerGRU",
    "dlinear": "DLinear",
    "s2s_lstm": "S2S-LSTM",
    "frets": "FreTS",
    "patchtst": "PatchTST",
}

MODEL_SOURCE = {
    "FITS": "FITSTest/duobu_1.py",
    "FITS-Delta": "baseline_experiments/run_fixed_parameter_fits.py",
    "NaiveLastValue": "baseline_experiments/run_baselines.py",
    "ARIMA": "baseline_experiments/run_baselines.py",
    "SingleLayerGRU": "baseline_experiments/run_baselines.py",
    "DLinear": "FITSTest/DLinear.py",
    "S2S-LSTM": "LSTM_D.py",
    "FreTS": "FITSTest/FreTS.py",
    "PatchTST": "FITSTest/7.py",
}

IMPLEMENTATION_VARIANT = {
    "FITS": "author_main_shared_complex_upsampler",
    "FITS-Delta": "author_fits_with_first_difference_residual_integration",
    "NaiveLastValue": "last_observation_repeated_for_all_horizons",
    "ARIMA": "per_channel_ARIMA_p_d_0_conditional_least_squares",
    "SingleLayerGRU": "one_layer_hidden_32_direct_multistep_head",
    "DLinear": "author_two_shared_linear_heads_with_batch_local_decomposition_fix",
    "S2S-LSTM": "author_encoder_decoder_hidden_128",
    "FreTS": "author_lightweight_hidden_12_truncated_frequency_variant",
    "PatchTST": "author_edge_sized_last_patch_head_variant",
}

RAW_COLUMNS = [
    "run_id",
    "dataset",
    "model",
    "seed",
    "source_file",
    "implementation_variant",
    "RMSE",
    "MAE",
    "R2",
    "MAPE_pct",
    "sMAPE_pct",
    "parameter_count",
    "model_state_bytes",
    "thop_MACs",
    "thop_note",
    "selection_seconds",
    "selected_orders",
    "best_epoch",
    "epochs_run",
    "best_val_MSE_scaled",
    "training_seconds",
    "training_device",
    "device_latency_median_ms",
    "device_latency_p95_ms",
    "cpu_latency_median_ms",
    "cpu_latency_p95_ms",
]

DETAILED_COLUMNS = [
    "run_id",
    "dataset",
    "model",
    "seed",
    "source_file",
    "implementation_variant",
    "scope",
    "channel",
    "horizon",
    "RMSE",
    "MAE",
    "R2",
    "MAPE_pct",
    "sMAPE_pct",
]


@dataclass(frozen=True)
class UnifiedConfig:
    nrows: int = 100_000
    seq_len: int = 100
    pred_len: int = 5
    moving_average_window: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 256
    eval_batch_size: int = 1024
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 7
    min_delta: float = 1e-7
    seeds: tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)
    inference_repeats: int = 1000
    inference_warmup: int = 100
    fits_cut_freq: int = 24
    gru_hidden_size: int = 32
    dlinear_kernel_size: int = 10
    s2s_hidden_size: int = 128
    frets_hidden_size: int = 12
    patch_len: int = 16
    patch_stride: int = 8
    patch_d_model: int = 64
    patch_heads: int = 4
    patch_layers: int = 2
    patch_d_ff: int = 128
    patch_dropout: float = 0.1
    arima_p_candidates: tuple[int, ...] = (1, 2, 5, 10)
    arima_d_candidates: tuple[int, ...] = (0, 1)


def base_config(cfg: UnifiedConfig):
    return BASE.ExperimentConfig(
        nrows=cfg.nrows,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        moving_average_window=cfg.moving_average_window,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        batch_size=cfg.batch_size,
        hidden_size=cfg.gru_hidden_size,
        learning_rate=cfg.learning_rate,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        seeds=cfg.seeds,
        arima_p_candidates=cfg.arima_p_candidates,
        arima_d_candidates=cfg.arima_d_candidates,
        inference_repeats=cfg.inference_repeats,
        fits_cut_freq=cfg.fits_cut_freq,
        fits_batch_size=cfg.batch_size,
        fits_epochs=cfg.max_epochs,
    )


class DeltaResidualFITS(nn.Module):
    """Forecast first differences with FITS and integrate from the last value."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, cut_freq: int):
        super().__init__()
        self.base = BASE.FITSModel(seq_len, pred_len, channels, cut_freq=cut_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        differences = torch.zeros_like(x)
        differences[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        predicted_differences = self.base(differences)
        return x[:, -1:, :] + torch.cumsum(predicted_differences, dim=1)


class DLinear(nn.Module):
    """Author DLinear architecture with sample-local decomposition.

    The original script flattened the batch before ``np.convolve``. That made a
    prediction depend on the other samples in the same mini-batch. This version
    keeps the author's two shared Linear(100, 5) heads and 10-point moving
    average, but performs decomposition independently for each sample/channel.
    """

    def __init__(self, seq_len: int, pred_len: int, channels: int, kernel_size: int = 10):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.kernel_size = kernel_size
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)

    def moving_average(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, L, C]. Asymmetric replicate padding preserves L for even k.
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        channel_first = x.permute(0, 2, 1)
        padded = F.pad(channel_first, (left, right), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1)
        return trend.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(x)
        seasonal = x - trend
        trend_out = self.trend_linear(trend.permute(0, 2, 1))
        seasonal_out = self.seasonal_linear(seasonal.permute(0, 2, 1))
        return (trend_out + seasonal_out).permute(0, 2, 1)


class Seq2SeqLSTM(nn.Module):
    """Encoder-decoder LSTM copied from LSTM_D.py."""

    def __init__(self, channels: int = 2, hidden_size: int = 128, pred_len: int = 5):
        super().__init__()
        # LSTM_D.py sets dropout=0.2 on a one-layer LSTM. PyTorch ignores that
        # argument when num_layers=1, so zero is the explicit equivalent here.
        self.encoder = nn.LSTM(channels, hidden_size, batch_first=True, dropout=0.0)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=0.0)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, channels))
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(x)
        decoder_input = hidden.repeat(self.pred_len, 1, 1).permute(1, 0, 2)
        outputs, _ = self.decoder(decoder_input, (hidden, cell))
        return self.fc(outputs)


class FreTS(nn.Module):
    """The lightweight FreTS variant implemented in FITSTest/FreTS.py."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        cut_freq: int = 24,
        hidden_size: int = 12,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.cut_freq = cut_freq
        self.freq_mlp = nn.Sequential(
            nn.Linear(cut_freq, hidden_size), nn.ReLU(), nn.Linear(hidden_size, cut_freq)
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, channels * 2),
        )
        self.predictor = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        normalized = (x - mean) / std
        spectrum = torch.fft.rfft(normalized, dim=1)[:, : self.cut_freq, :]
        real_imag = torch.cat([spectrum.real, spectrum.imag], dim=-1)
        mixed_frequency = self.freq_mlp(real_imag.permute(0, 2, 1)).permute(0, 2, 1)
        mixed_channel = self.channel_mlp(mixed_frequency)
        output_ri = real_imag + mixed_channel
        output_spectrum = torch.complex(
            output_ri[..., : self.channels], output_ri[..., self.channels :]
        )
        full_spectrum = torch.zeros(
            x.size(0),
            self.seq_len // 2 + 1,
            self.channels,
            dtype=torch.cfloat,
            device=x.device,
        )
        full_spectrum[:, : self.cut_freq, :] = output_spectrum
        time_signal = torch.fft.irfft(full_spectrum, n=self.seq_len, dim=1)
        output = self.predictor(time_signal.permute(0, 2, 1)).permute(0, 2, 1)
        return output * std + mean


class RevIN(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(channels))
            self.affine_bias = nn.Parameter(torch.zeros(channels))

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + self.eps).detach()
        normalized = (x - mean) / stdev
        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias
        return normalized, mean, stdev

    def denormalize(self, x: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * stdev + mean


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

    def forward(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=stride)
        return self.value_embedding(patches)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.d_k = d_model // heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, patches, dim = x.shape
        q = self.w_q(x).view(batch, patches, self.heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, patches, self.heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, patches, self.heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch, patches, dim)
        return self.w_o(output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.ff(self.norm2(x)))


class PatchTST(nn.Module):
    """The author's current edge-sized PatchTST variant from FITSTest/7.py."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        patch_len: int,
        stride: int,
        d_model: int,
        heads: int,
        layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.channels = channels
        self.pred_len = pred_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.revin = RevIN(channels)
        self.patch_embedding = PatchEmbedding(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.encoder = nn.ModuleList(
            [TransformerBlock(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        normalized, mean, stdev = self.revin.normalize(x)
        encoded = normalized.permute(0, 2, 1).reshape(batch * channels, length)
        encoded = self.patch_embedding(encoded, self.stride)
        encoded = self.dropout(encoded + self.pos_embedding)
        for layer in self.encoder:
            encoded = layer(encoded)
        representation = self.norm(encoded)[:, -1, :].view(batch, channels, -1)
        output = self.head(representation).permute(0, 2, 1)
        return self.revin.denormalize(output, mean, stdev)


def make_model(model_key: str, cfg: UnifiedConfig, channels: int) -> nn.Module:
    if model_key == "fits":
        return BASE.FITSModel(cfg.seq_len, cfg.pred_len, channels, cut_freq=cfg.fits_cut_freq)
    if model_key == "fits_delta":
        return DeltaResidualFITS(cfg.seq_len, cfg.pred_len, channels, cfg.fits_cut_freq)
    if model_key == "gru":
        return BASE.SingleLayerGRU(channels, cfg.gru_hidden_size, cfg.pred_len)
    if model_key == "dlinear":
        return DLinear(cfg.seq_len, cfg.pred_len, channels, cfg.dlinear_kernel_size)
    if model_key == "s2s_lstm":
        return Seq2SeqLSTM(channels, cfg.s2s_hidden_size, cfg.pred_len)
    if model_key == "frets":
        return FreTS(
            cfg.seq_len,
            cfg.pred_len,
            channels,
            cfg.fits_cut_freq,
            cfg.frets_hidden_size,
        )
    if model_key == "patchtst":
        return PatchTST(
            cfg.seq_len,
            cfg.pred_len,
            channels,
            cfg.patch_len,
            cfg.patch_stride,
            cfg.patch_d_model,
            cfg.patch_heads,
            cfg.patch_layers,
            cfg.patch_d_ff,
            cfg.patch_dropout,
        )
    raise KeyError(model_key)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_state_bytes(model: nn.Module) -> int:
    return int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))


def profile_macs(model: nn.Module, sample: torch.Tensor) -> tuple[float, str]:
    """Return THOP MACs with an explicit limitation note.

    THOP does not count FFT/IFFT or the explicit attention matrix products, so
    the value is useful only as a reproducible lower-bound-style implementation
    profile, not as a complete analytical operation count.
    """

    try:
        from thop import profile

        copied = copy.deepcopy(model).cpu().eval()
        macs, _ = profile(copied, inputs=(sample.cpu(),), verbose=False)
        return float(macs), "THOP; excludes unsupported FFT/IFFT and explicit attention matmul operations"
    except Exception as exc:  # pragma: no cover - environment dependent
        return float("nan"), f"THOP failed: {type(exc).__name__}: {exc}"


def timed_inference(model: nn.Module, sample: torch.Tensor, repeats: int, warmup: int) -> dict:
    model.eval()

    def call() -> None:
        with torch.inference_mode():
            model(sample)
        if sample.device.type == "cuda":
            torch.cuda.synchronize(sample.device)

    timing = BASE.percentile_timing(call, repeats=repeats, warmup=warmup)
    return timing


def evaluate_scaled(
    model: nn.Module,
    loader: DataLoader,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for x_batch, y_batch in loader:
            predictions.append(model(x_batch.to(device, non_blocking=True)).cpu().numpy())
            targets.append(y_batch.numpy())
    predicted_scaled = np.concatenate(predictions)
    target_scaled = np.concatenate(targets)
    predicted = predicted_scaled * std[None, None, :] + mean[None, None, :]
    target = target_scaled * std[None, None, :] + mean[None, None, :]
    return target.astype(np.float32), predicted.astype(np.float32)


def train_learning_model(
    model_key: str,
    splits: dict[str, np.ndarray],
    cfg: UnifiedConfig,
    seed: int,
    dataset_dir: Path,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    BASE.set_seed(seed)
    mean = splits["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = splits["train"].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0

    def scale(values: np.ndarray) -> np.ndarray:
        return ((values - mean) / std).astype(np.float32)

    windows = {
        name: BASE.make_windows(scale(values), cfg.seq_len, cfg.pred_len)
        for name, values in splits.items()
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        BASE.WindowDataset(*windows["train"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        BASE.WindowDataset(*windows["val"]),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        BASE.WindowDataset(*windows["test"]),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(model_key, cfg, splits["train"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []
    training_start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_sum = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * len(x_batch)
        train_mse = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.inference_mode():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                val_sum += float(criterion(model(x_batch), y_batch).item()) * len(x_batch)
        val_mse = val_sum / len(val_loader.dataset)
        history.append({"epoch": epoch, "train_MSE_scaled": train_mse, "val_MSE_scaled": val_mse})
        if val_mse < best_val - cfg.min_delta:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"{MODEL_LABELS[model_key]} seed={seed} epoch={epoch:02d} "
            f"train={train_mse:.6f} val={val_mse:.6f} best={best_val:.6f} "
            f"patience={bad_epochs}/{cfg.patience}",
            flush=True,
        )
        if bad_epochs >= cfg.patience:
            break
    training_seconds = time.perf_counter() - training_start
    if best_state is None:
        raise RuntimeError(f"{model_key} produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    y_true, y_pred = evaluate_scaled(model, test_loader, mean, std, device)

    x_test_scaled = windows["test"][0]
    sample_device = torch.from_numpy(x_test_scaled[:1]).to(device)
    device_timing = timed_inference(
        model, sample_device, cfg.inference_repeats, cfg.inference_warmup
    )
    cpu_model = copy.deepcopy(model).cpu().eval()
    sample_cpu = torch.from_numpy(x_test_scaled[:1])
    cpu_timing = timed_inference(
        cpu_model, sample_cpu, cfg.inference_repeats, cfg.inference_warmup
    )
    macs, mac_note = profile_macs(cpu_model, sample_cpu)

    label = MODEL_LABELS[model_key]
    checkpoint = {
        "model_state": best_state,
        "model_key": model_key,
        "model_label": label,
        "seed": seed,
        "mean": mean,
        "std": std,
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "source_file": MODEL_SOURCE[label],
        "implementation_variant": IMPLEMENTATION_VARIANT[label],
    }
    torch.save(checkpoint, dataset_dir / f"{model_key}_seed_{seed}.pt")
    result = {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_MSE_scaled": best_val,
        "training_seconds": training_seconds,
        "parameter_count": count_parameters(model),
        "model_state_bytes": model_state_bytes(model),
        "thop_MACs": macs,
        "thop_note": mac_note,
        "training_device": str(device),
        "device_latency_median_ms": device_timing["latency_median_ms"],
        "device_latency_p95_ms": device_timing["latency_p95_ms"],
        "cpu_latency_median_ms": cpu_timing["latency_median_ms"],
        "cpu_latency_p95_ms": cpu_timing["latency_p95_ms"],
    }
    return result, y_true, y_pred, history


def append_row(path: Path, row: dict) -> None:
    columns = DETAILED_COLUMNS if "detailed" in path.name else RAW_COLUMNS
    normalized = {column: row.get(column, np.nan) for column in columns}
    pd.DataFrame([normalized], columns=columns).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["RMSE", "MAE", "R2", "MAPE_pct", "sMAPE_pct"]
    efficiency = [
        "parameter_count",
        "model_state_bytes",
        "thop_MACs",
        "training_seconds",
        "best_epoch",
        "cpu_latency_median_ms",
        "cpu_latency_p95_ms",
        "device_latency_median_ms",
        "device_latency_p95_ms",
    ]
    records = []
    for (dataset, model), group in raw.groupby(["dataset", "model"], sort=False):
        row = {
            "dataset": dataset,
            "model": model,
            "runs": len(group),
            "source_file": group["source_file"].iloc[0],
            "implementation_variant": group["implementation_variant"].iloc[0],
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        for metric in efficiency:
            if metric in group:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
        records.append(row)
    order = {label: index for index, label in enumerate(MODEL_LABELS.values())}
    result = pd.DataFrame(records)
    result["_order"] = result["model"].map(order)
    return result.sort_values(["dataset", "_order"]).drop(columns="_order").reset_index(drop=True)


def environment_info() -> dict:
    return {
        **BASE.environment_info(),
        "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "hostname": platform.node(),
        "runner": str(Path(__file__).resolve()),
    }


def save_prediction(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    np.savez_compressed(path, y_true=y_true, y_pred=y_pred)


def existing_ids(raw_path: Path) -> set[str]:
    if not raw_path.exists():
        return set()
    frame = pd.read_csv(raw_path, usecols=["run_id"])
    return set(frame["run_id"].astype(str))


def canonical_target_check(canonical: np.ndarray, candidate: np.ndarray, context: str) -> None:
    if canonical.shape != candidate.shape or not np.allclose(canonical, candidate, atol=2e-5, rtol=1e-6):
        difference = float(np.max(np.abs(canonical - candidate))) if canonical.shape == candidate.shape else math.inf
        raise AssertionError(f"Test targets differ for {context}; max_abs_difference={difference}")


def run_dataset(
    dataset_name: str,
    model_keys: list[str],
    cfg: UnifiedConfig,
    output: Path,
    raw_path: Path,
    detailed_path: Path,
    completed: set[str],
) -> dict:
    spec = BASE.DATASETS[dataset_name]
    print(f"\n===== DATASET {dataset_name} =====", flush=True)
    dataset_dir = output / dataset_name.replace("/", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = BASE.load_dataset(spec, base_config(cfg))
    splits = BASE.split_series(data, base_config(cfg))
    windows = {
        name: BASE.make_windows(values, cfg.seq_len, cfg.pred_len)
        for name, values in splits.items()
    }
    canonical_y = windows["test"][1]
    np.savez_compressed(dataset_dir / "canonical_test_targets.npz", y_true=canonical_y)
    metadata["split_rows"] = {name: int(len(values)) for name, values in splits.items()}
    metadata["window_counts"] = {name: int(len(x_y[0])) for name, x_y in windows.items()}
    metadata["canonical_test_target_sha256"] = BASE.hashlib.sha256(canonical_y.tobytes()).hexdigest()
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if "naive" in model_keys:
        run_id = f"{dataset_name}|naive|deterministic"
        if run_id not in completed:
            prediction = BASE.naive_predict(windows["test"][0], cfg.pred_len)
            result = BASE.metrics(canonical_y, prediction)
            timing = BASE.percentile_timing(
                lambda: BASE.naive_predict(windows["test"][0][:1], cfg.pred_len),
                cfg.inference_repeats,
                cfg.inference_warmup,
            )
            label = MODEL_LABELS["naive"]
            row = {
                "run_id": run_id,
                "dataset": dataset_name,
                "model": label,
                "seed": "deterministic",
                "source_file": MODEL_SOURCE[label],
                "implementation_variant": IMPLEMENTATION_VARIANT[label],
                **result,
                "parameter_count": 0,
                "model_state_bytes": 0,
                "cpu_latency_median_ms": timing["latency_median_ms"],
                "cpu_latency_p95_ms": timing["latency_p95_ms"],
            }
            append_row(raw_path, row)
            for detail in BASE.detailed_metrics(canonical_y, prediction):
                append_row(detailed_path, {"run_id": run_id, **row, **detail})
            save_prediction(dataset_dir / "naive_predictions.npz", canonical_y, prediction)
            completed.add(run_id)
            print(f"Naive: {result}", flush=True)

    if "arima" in model_keys:
        run_id = f"{dataset_name}|arima|deterministic"
        if run_id not in completed:
            selection_start = time.perf_counter()
            orders, coefficients, search = BASE.select_arima_orders(
                splits["train"], windows["val"][0], windows["val"][1], base_config(cfg)
            )
            selection_seconds = time.perf_counter() - selection_start
            pd.DataFrame(search).to_csv(dataset_dir / "arima_validation_search.csv", index=False)
            prediction = BASE.predict_arima_p_d_0(
                windows["test"][0], orders, coefficients, cfg.pred_len
            )
            result = BASE.metrics(canonical_y, prediction)
            timing = BASE.percentile_timing(
                lambda: BASE.predict_arima_p_d_0(
                    windows["test"][0][:1], orders, coefficients, cfg.pred_len
                ),
                cfg.inference_repeats,
                cfg.inference_warmup,
            )
            parameter_count = int(sum(1 + len(phi) for _, phi in coefficients))
            label = MODEL_LABELS["arima"]
            row = {
                "run_id": run_id,
                "dataset": dataset_name,
                "model": label,
                "seed": "deterministic",
                "source_file": MODEL_SOURCE[label],
                "implementation_variant": IMPLEMENTATION_VARIANT[label],
                **result,
                "parameter_count": parameter_count,
                "selection_seconds": selection_seconds,
                "selected_orders": json.dumps([(p, d, 0) for p, d in orders]),
                "cpu_latency_median_ms": timing["latency_median_ms"],
                "cpu_latency_p95_ms": timing["latency_p95_ms"],
            }
            append_row(raw_path, row)
            for detail in BASE.detailed_metrics(canonical_y, prediction):
                append_row(detailed_path, {"run_id": run_id, **row, **detail})
            save_prediction(dataset_dir / "arima_predictions.npz", canonical_y, prediction)
            (dataset_dir / "arima_model.json").write_text(
                json.dumps(
                    {
                        "estimator": "conditional least squares",
                        "orders": [(p, d, 0) for p, d in orders],
                        "coefficients": [
                            {"intercept": intercept, "phi": phi.tolist()}
                            for intercept, phi in coefficients
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            completed.add(run_id)
            print(f"ARIMA orders={[(p, d, 0) for p, d in orders]}: {result}", flush=True)

    learning_keys = [
        key for key in model_keys if key not in {"naive", "arima"}
    ]
    for model_key in learning_keys:
        for seed in cfg.seeds:
            run_id = f"{dataset_name}|{model_key}|{seed}"
            if run_id in completed:
                print(f"SKIP completed {run_id}", flush=True)
                continue
            timing, y_true, prediction, history = train_learning_model(
                model_key, splits, cfg, seed, dataset_dir
            )
            canonical_target_check(canonical_y, y_true, run_id)
            result = BASE.metrics(canonical_y, prediction)
            label = MODEL_LABELS[model_key]
            row = {
                "run_id": run_id,
                "dataset": dataset_name,
                "model": label,
                "seed": seed,
                "source_file": MODEL_SOURCE[label],
                "implementation_variant": IMPLEMENTATION_VARIANT[label],
                **result,
                **timing,
            }
            append_row(raw_path, row)
            for detail in BASE.detailed_metrics(canonical_y, prediction):
                append_row(detailed_path, {"run_id": run_id, **row, **detail})
            pd.DataFrame(history).to_csv(
                dataset_dir / f"{model_key}_seed_{seed}_history.csv", index=False
            )
            save_prediction(
                dataset_dir / f"{model_key}_seed_{seed}_predictions.npz",
                canonical_y,
                prediction,
            )
            completed.add(run_id)
            print(f"RESULT {run_id}: {result}", flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified nine-model manuscript prediction benchmark")
    parser.add_argument("--datasets", nargs="+", choices=list(BASE.DATASETS), default=list(BASE.DATASETS))
    parser.add_argument("--models", nargs="+", choices=list(MODEL_LABELS), default=list(MODEL_LABELS))
    parser.add_argument("--nrows", type=int, default=100_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--inference-repeats", type=int, default=1000)
    parser.add_argument("--inference-warmup", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "all_models_unified_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = UnifiedConfig(
        nrows=args.nrows,
        seeds=tuple(args.seeds),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        inference_repeats=args.inference_repeats,
        inference_warmup=args.inference_warmup,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    requested_config = {
        "config": asdict(cfg),
        "datasets": args.datasets,
        "models": args.models,
        "model_labels": MODEL_LABELS,
        "model_sources": MODEL_SOURCE,
        "implementation_variants": IMPLEMENTATION_VARIANT,
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != requested_config:
            raise RuntimeError("Existing output directory has a different configuration; use a new --output")
    else:
        config_path.write_text(
            json.dumps(requested_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "environment.json").write_text(
        json.dumps(environment_info(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raw_path = output / "prediction_metrics_raw.csv"
    detailed_path = output / "prediction_metrics_detailed_raw.csv"
    completed = existing_ids(raw_path)
    metadata = {}
    run_start = time.perf_counter()
    run_info_path = output / "run_info.json"
    run_info_path.write_text(
        json.dumps(
            {
                "status": "running",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "already_completed_runs": len(completed),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        for dataset_name in args.datasets:
            metadata[dataset_name] = run_dataset(
                dataset_name,
                args.models,
                cfg,
                output,
                raw_path,
                detailed_path,
                completed,
            )
            raw = pd.read_csv(raw_path)
            summarize(raw).to_csv(output / "prediction_summary_mean_std.csv", index=False)
        raw = pd.read_csv(raw_path)
        summary = summarize(raw)
        summary.to_csv(output / "prediction_summary_mean_std.csv", index=False)
        efficiency_columns = [
            column
            for column in summary.columns
            if column
            in {
                "dataset",
                "model",
                "runs",
                "parameter_count_mean",
                "model_state_bytes_mean",
                "thop_MACs_mean",
                "training_seconds_mean",
                "best_epoch_mean",
                "cpu_latency_median_ms_mean",
                "cpu_latency_p95_ms_mean",
                "device_latency_median_ms_mean",
                "device_latency_p95_ms_mean",
                "source_file",
                "implementation_variant",
            }
        ]
        summary[efficiency_columns].to_csv(output / "efficiency_metrics.csv", index=False)
        (output / "dataset_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        expected_runs = len(args.datasets) * (
            sum(1 for key in args.models if key in {"naive", "arima"})
            + len(cfg.seeds) * sum(1 for key in args.models if key not in {"naive", "arima"})
        )
        run_info = {
            "status": "complete",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds_this_invocation": time.perf_counter() - run_start,
            "completed_runs": int(len(raw)),
            "expected_runs": int(expected_runs),
            "datasets": args.datasets,
            "models": args.models,
        }
        run_info_path.write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if len(raw) != expected_runs:
            raise AssertionError(f"Expected {expected_runs} runs, found {len(raw)}")
        print(f"\nCOMPLETE: {output}", flush=True)
    except Exception as exc:
        run_info_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds_this_invocation": time.perf_counter() - run_start,
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_runs": len(completed),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
