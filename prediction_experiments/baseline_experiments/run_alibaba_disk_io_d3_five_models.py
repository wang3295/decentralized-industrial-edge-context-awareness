from __future__ import annotations

"""Five-model predictor comparison for the Alibaba disk-I/O d=3 study.

This script reuses the exact prepared 20-machine data, chronological splits,
window protocol, seeds, optimization settings, and latency protocol from the
formal disk-I/O experiment.  It adds FITS, DLinear, and a single-layer GRU to
the existing FITS-Delta and Naive comparison without rerunning state updates.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D3 = load_module("disk_d3_five_model_shared", HERE / "run_alibaba_network_throughput_d3.py")
UNIFIED = D3.UNIFIED

CHANNELS = ("CPU", "Memory", "DiskIO")
MODEL_ORDER = ("NaiveLastValue", "FITS", "DLinear", "SingleLayerGRU", "FITS-Delta")
TRAINED_MODELS = {
    "fits": "FITS",
    "dlinear": "DLinear",
    "gru": "SingleLayerGRU",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_batches(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    original_shape = x.shape[:2]
    flat = x.reshape(-1, *x.shape[2:])
    loader = DataLoader(TensorDataset(torch.from_numpy(flat)), batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            outputs.append(model(batch).cpu().numpy())
    prediction = np.concatenate(outputs, axis=0)
    return prediction.reshape(*original_shape, *prediction.shape[1:]).astype(np.float32)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    row: dict[str, float] = {}
    channel_rmse: list[float] = []
    for index, channel in enumerate(CHANNELS):
        metrics = UNIFIED.BASE.metrics(y_true[..., index : index + 1], y_pred[..., index : index + 1])
        prefix = channel.lower()
        row[f"{prefix}_rmse"] = float(metrics["RMSE"])
        row[f"{prefix}_mae"] = float(metrics["MAE"])
        row[f"{prefix}_r2"] = float(metrics["R2"])
        channel_rmse.append(float(metrics["RMSE"]))
    global_metrics = UNIFIED.BASE.metrics(y_true, y_pred)
    row["global_rmse"] = float(global_metrics["RMSE"])
    row["global_mae"] = float(global_metrics["MAE"])
    row["global_r2"] = float(global_metrics["R2"])
    row["diskio_nrmse"] = row["diskio_rmse"]
    row["macro_nrmse"] = float(np.mean(channel_rmse))
    return row


def train_or_load(
    model_key: str,
    seed: int,
    windows: dict[str, tuple[np.ndarray, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
    output: Path,
    cfg: UNIFIED.UnifiedConfig,
) -> tuple[nn.Module, dict[str, object], np.ndarray]:
    checkpoint_path = output / f"{model_key}_d3_seed_{seed}.pt"
    model = UNIFIED.make_model(model_key, cfg, len(CHANNELS)).cpu()
    train_x, train_y = windows["train"]
    val_x, val_y = windows["val"]
    test_x, _ = windows["test"]
    train_x_s = D3.scaled(train_x, mean, std).reshape(-1, cfg.seq_len, len(CHANNELS))
    train_y_s = D3.scaled(train_y, mean, std).reshape(-1, cfg.pred_len, len(CHANNELS))
    val_x_s = D3.scaled(val_x, mean, std)
    val_y_s = D3.scaled(val_y, mean, std)
    test_x_s = D3.scaled(test_x, mean, std)

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
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()
        best_loss = math.inf
        best_state = None
        best_epoch = 0
        bad_epochs = 0
        history: list[dict[str, float | int]] = []
        started = time.perf_counter()
        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            train_sum = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                train_sum += float(loss.item()) * len(x_batch)
            model.eval()
            val_prediction_s = predict_batches(model, val_x_s, cfg.eval_batch_size)
            val_loss = float(np.mean(np.square(val_prediction_s - val_y_s)))
            train_loss = train_sum / len(loader.dataset)
            history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_mse_scaled": val_loss})
            print(
                f"{TRAINED_MODELS[model_key]} d3 seed={seed} epoch={epoch:02d} "
                f"train_MSE={train_loss:.7f} val_MSE={val_loss:.7f}",
                flush=True,
            )
            if val_loss < best_loss - cfg.min_delta:
                best_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
        if best_state is None:
            raise RuntimeError(f"{model_key} produced no valid checkpoint")
        model.load_state_dict(best_state)
        training_meta = {
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "best_validation_mse_scaled": best_loss,
            "training_seconds": time.perf_counter() - started,
        }
        pd.DataFrame(history).to_csv(output / f"{model_key}_d3_seed_{seed}_history.csv", index=False)
        torch.save(
            {
                "model_state": best_state,
                "mean": mean,
                "std": std,
                "channels": CHANNELS,
                "training_meta": training_meta,
                "model_key": model_key,
                "model_label": TRAINED_MODELS[model_key],
                "config": asdict(cfg),
            },
            checkpoint_path,
        )

    model.eval()
    test_prediction = predict_batches(model, test_x_s, cfg.eval_batch_size) * std + mean
    sample = torch.from_numpy(test_x_s.reshape(-1, cfg.seq_len, len(CHANNELS))[:1])
    timing = UNIFIED.timed_inference(
        model, sample, repeats=cfg.inference_repeats, warmup=cfg.inference_warmup
    )
    training_meta.update(timing)
    training_meta["parameter_count"] = UNIFIED.count_parameters(model)
    training_meta["model_state_bytes"] = UNIFIED.model_state_bytes(model)
    np.savez_compressed(
        output / f"{model_key}_d3_seed_{seed}_predictions.npz",
        test_prediction=test_prediction.astype(np.float32),
    )
    return model, training_meta, test_prediction.astype(np.float32)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in raw.columns if column not in {"model", "seed"}]
    summary = raw.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index().fillna(0.0)
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    summary["_order"] = summary["model"].map(order)
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def per_machine_rows(
    model: str,
    seed: int | str,
    machine_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "seed": seed,
            "machine_id": str(machine_id),
            **metric_row(y_true[index], y_pred[index]),
        }
        for index, machine_id in enumerate(machine_ids)
    ]


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_tests(per_machine: pd.DataFrame) -> pd.DataFrame:
    # Average stochastic seeds within each machine, then compare the same 20
    # machines.  This prevents five training seeds from being counted as 100
    # independent machines.
    averaged = per_machine.groupby(["model", "machine_id"], as_index=False)["global_rmse"].mean()
    pivot = averaged.pivot(index="machine_id", columns="model", values="global_rmse")
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for baseline in MODEL_ORDER[:-1]:
        differences = pivot[baseline] - pivot["FITS-Delta"]
        test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        p_value = float(test.pvalue)
        p_values.append(p_value)
        rows.append(
            {
                "comparison": f"FITS-Delta vs {baseline}",
                "paired_machines": len(differences),
                "median_baseline_minus_fits_delta_global_rmse": float(np.median(differences)),
                "wilcoxon_statistic": float(test.statistic),
                "p_raw": p_value,
            }
        )
    adjusted = holm_adjust(p_values)
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
        row["significant_0_05"] = bool(value < 0.05)
    return pd.DataFrame(rows)


def write_report(output: Path, summary: pd.DataFrame, tests: pd.DataFrame, manifest: dict[str, object]) -> None:
    rows = summary.set_index("model").loc[list(MODEL_ORDER)].reset_index()
    lines = [
        "# Alibaba磁盘I/O三维五模型预测对比",
        "",
        "## 实验定位",
        "",
        "该实验是CPU、内存和磁盘I/O三维资源状态的泛化性验证。它复用正式磁盘I/O实验的20台机器、时间窗口和70%/15%/15%时间划分，不替代正文中的三数据集九模型主实验。",
        "",
        "## 统一设置",
        "",
        "- 输入100步，预测5步；所有窗口均在单台机器内部构造。",
        "- FITS、DLinear、单层GRU和FITS-Delta均运行5个独立种子；Naive为确定性方法。",
        "- 所有学习模型使用相同的训练/验证/测试数据和早停规则，测试集不参与模型选择。",
        "- CPU推理时延为当前运行主机上的batch size=1、预热100次、重复1000次结果，不应标注为工控机时延。",
        "",
        "## 五模型结果",
        "",
        "| 模型 | Disk-I/O RMSE↓ | Disk-I/O R²↑ | 三维总体R²↑ | Macro NRMSE↓ | 参数量 | CPU中位/P95时延(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row.model} | {row.diskio_rmse_mean:.5f} | {row.diskio_r2_mean:.4f} | "
            f"{row.global_r2_mean:.4f} | {row.macro_nrmse_mean:.5f} | "
            f"{int(round(row.parameter_count_mean))} | "
            f"{row.latency_median_ms_mean:.4f}/{row.latency_p95_ms_mean:.4f} |"
        )
    lines.extend([
        "",
        "## 配对统计检验",
        "",
        "以机器为独立统计单位，先在每台机器内对训练种子取平均，再对20台相同机器的总体RMSE执行双侧Wilcoxon符号秩检验，并采用Holm方法校正四次比较。",
        "",
        "| 比较 | 中位RMSE差（基线−FITS-Delta） | 原始p | Holm校正p | 显著 |",
        "|---|---:|---:|---:|:---:|",
    ])
    for _, row in tests.iterrows():
        lines.append(
            f"| {row.comparison} | {row.median_baseline_minus_fits_delta_global_rmse:.6f} | "
            f"{row.p_raw:.3g} | {row.p_holm:.3g} | {'是' if row.significant_0_05 else '否'} |"
        )
    lines.extend([
        "",
        "## 结果边界",
        "",
        "该五模型表用于证明FITS-Delta在新增磁盘I/O维度后，相对于持久性、原始FITS、轻量线性模型和经典循环模型仍具有可比较的精度—复杂度表现。论文中不应把它表述为完整九模型重新选型，也不应声称时延来自工控机。",
        "",
        f"数据文件SHA-256：`{manifest['data_sha256']}`。",
        "",
    ])
    (output / "Alibaba磁盘IO三维五模型预测结果.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=HERE / "data" / "processed" / "alibaba_disk_io_d3" / "alibaba_disk_io_d3.npz",
    )
    parser.add_argument(
        "--existing-dir",
        type=Path,
        default=HERE / "results" / "alibaba_disk_io_d3_formal_20260915",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "alibaba_disk_io_d3_five_models_20260915",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--inference-repeats", type=int, default=1000)
    parser.add_argument("--inference-warmup", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    saved = np.load(args.data, allow_pickle=False)
    values = saved["values"].astype(np.float32)
    machine_ids = saved["machine_ids"]
    if values.shape != (20, 5000, 3):
        raise AssertionError(f"Unexpected prepared data shape: {values.shape}")
    D3.CHANNELS = CHANNELS
    windows, mean, std = D3.make_split_windows(values, 100, 5, 0.70, 0.15)
    test_x, test_y = windows["test"]
    cfg = UNIFIED.UnifiedConfig(
        seq_len=100,
        pred_len=5,
        moving_average_window=5,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seeds=tuple(args.seeds),
        inference_repeats=args.inference_repeats,
        inference_warmup=args.inference_warmup,
        fits_cut_freq=24,
        gru_hidden_size=32,
        dlinear_kernel_size=10,
    )

    prediction_rows: list[dict[str, object]] = []
    machine_rows: list[dict[str, object]] = []

    naive_prediction = np.repeat(test_x[:, :, -1:, :], cfg.pred_len, axis=2)
    naive_samples: list[float] = []
    for index in range(cfg.inference_warmup + cfg.inference_repeats):
        started = time.perf_counter_ns()
        _ = np.repeat(test_x[0, 0, -1:, :], cfg.pred_len, axis=0)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if index >= cfg.inference_warmup:
            naive_samples.append(elapsed)
    naive_row: dict[str, object] = {
        "model": "NaiveLastValue",
        "seed": "deterministic",
        **metric_row(test_y, naive_prediction),
        "parameter_count": 0,
        "model_state_bytes": 0,
        "latency_median_ms": float(np.median(naive_samples)),
        "latency_p95_ms": float(np.percentile(naive_samples, 95)),
        "training_seconds": 0.0,
        "best_epoch": 0,
    }
    prediction_rows.append(naive_row)
    machine_rows.extend(per_machine_rows("NaiveLastValue", "deterministic", machine_ids, test_y, naive_prediction))
    np.savez_compressed(args.output / "naive_d3_predictions.npz", test_prediction=naive_prediction)

    for model_key, model_label in TRAINED_MODELS.items():
        for seed in args.seeds:
            _, meta, prediction = train_or_load(
                model_key, seed, windows, mean, std, args.output, cfg
            )
            prediction_rows.append(
                {
                    "model": model_label,
                    "seed": seed,
                    **metric_row(test_y, prediction),
                    "parameter_count": meta["parameter_count"],
                    "model_state_bytes": meta["model_state_bytes"],
                    "latency_median_ms": meta["latency_median_ms"],
                    "latency_p95_ms": meta["latency_p95_ms"],
                    "training_seconds": meta["training_seconds"],
                    "best_epoch": meta["best_epoch"],
                }
            )
            machine_rows.extend(per_machine_rows(model_label, seed, machine_ids, test_y, prediction))

    existing_raw = pd.read_csv(args.existing_dir / "prediction_raw.csv")
    fits_delta_raw = existing_raw.loc[existing_raw["model"] == "FITS-Delta"].copy()
    for _, old_row in fits_delta_raw.iterrows():
        seed = int(old_row["seed"])
        if seed not in args.seeds:
            continue
        pred_path = args.existing_dir / f"fits_delta_d3_seed_{seed}_predictions.npz"
        prediction = np.load(pred_path, allow_pickle=False)["test_prediction"].astype(np.float32)
        recomputed = metric_row(test_y, prediction)
        for key, value in recomputed.items():
            if not np.isclose(value, float(old_row[key]), rtol=1e-6, atol=1e-8):
                raise AssertionError(f"Existing FITS-Delta mismatch for seed={seed}, metric={key}")
        prediction_rows.append(old_row.to_dict())
        machine_rows.extend(per_machine_rows("FITS-Delta", seed, machine_ids, test_y, prediction))

    raw = pd.DataFrame(prediction_rows)
    per_machine = pd.DataFrame(machine_rows)
    expected_runs = 1 + len(args.seeds) * 4
    if len(raw) != expected_runs or set(raw["model"]) != set(MODEL_ORDER):
        raise AssertionError(f"Expected {expected_runs} five-model records, got {len(raw)}")
    numeric = raw.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise AssertionError("Non-finite predictor metric detected")
    summary = summarize(raw)
    tests = paired_tests(per_machine)
    raw.to_csv(args.output / "prediction_raw.csv", index=False)
    summary.to_csv(args.output / "prediction_summary.csv", index=False)
    per_machine.to_csv(args.output / "prediction_machine_level.csv", index=False)
    tests.to_csv(args.output / "paired_wilcoxon_holm.csv", index=False)
    manifest = {
        "purpose": "five-model d=3 predictor comparison with Disk I/O",
        "models": list(MODEL_ORDER),
        "trained_models": list(TRAINED_MODELS.values()) + ["FITS-Delta"],
        "seeds": args.seeds,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_shape": list(values.shape),
        "machine_count": len(machine_ids),
        "machine_ids": machine_ids.tolist(),
        "split": "chronological 70/15/15 within each machine",
        "no_cross_machine_windows": True,
        "config": asdict(cfg),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "source_fits_delta_dir": str(args.existing_dir.resolve()),
        "record_count": len(raw),
        "per_machine_record_count": len(per_machine),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.output, summary, tests, manifest)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved five-model results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
