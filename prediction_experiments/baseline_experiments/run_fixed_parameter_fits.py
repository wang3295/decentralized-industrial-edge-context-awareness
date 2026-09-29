from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run_baselines", HERE / "run_baselines.py")
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


VARIANTS = {
    "trainopt": {"model": "standard", "loss": "mse"},
    "balanced": {"model": "standard", "loss": "balanced_mse"},
    "huber": {"model": "standard", "loss": "standardized_huber"},
    "delta": {"model": "delta", "loss": "mse"},
    "delta_balanced": {"model": "delta", "loss": "balanced_mse"},
}


class DeltaResidualFITS(nn.Module):
    """Forecast first differences with FITS, then integrate from the latest value."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, cut_freq: int):
        super().__init__()
        self.base = BASE.FITSModel(seq_len, pred_len, channels, cut_freq=cut_freq, individual=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        differences = torch.zeros_like(x)
        differences[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        predicted_differences = self.base(differences)
        return x[:, -1:, :] + torch.cumsum(predicted_differences, dim=1)


def make_model(variant: str, seq_len: int, pred_len: int, channels: int, cut_freq: int) -> nn.Module:
    if VARIANTS[variant]["model"] == "delta":
        return DeltaResidualFITS(seq_len, pred_len, channels, cut_freq)
    return BASE.FITSModel(seq_len, pred_len, channels, cut_freq=cut_freq, individual=False)


def training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    channel_std: torch.Tensor,
) -> torch.Tensor:
    if loss_name == "mse":
        return torch.mean((prediction - target) ** 2)
    standardized_error = (prediction - target) / channel_std.view(1, 1, -1)
    if loss_name == "balanced_mse":
        return torch.mean(standardized_error**2)
    if loss_name == "standardized_huber":
        return nn.functional.smooth_l1_loss(standardized_error, torch.zeros_like(standardized_error), beta=1.0)
    raise ValueError(loss_name)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for xb, yb in loader:
            predictions.append(model(xb.to(device, non_blocking=True)).cpu().numpy())
            targets.append(yb.numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def train_one(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray | None,
    variant: str,
    cfg,
    seed: int,
    cut_freq: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[dict, np.ndarray | None, np.ndarray | None, dict]:
    BASE.set_seed(seed)
    x_train, y_train = BASE.make_windows(train_data, cfg.seq_len, cfg.pred_len)
    x_val, y_val = BASE.make_windows(val_data, cfg.seq_len, cfg.pred_len)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        BASE.WindowDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        BASE.WindowDataset(x_val, y_val), batch_size=max(batch_size, 512), shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(variant, cfg.seq_len, cfg.pred_len, train_data.shape[1], cut_freq).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    channel_std = torch.from_numpy(train_data.std(axis=0, dtype=np.float64).astype(np.float32)).to(device)
    channel_std[channel_std < 1e-6] = 1.0
    best_val_mse = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = training_loss(prediction, yb, VARIANTS[variant]["loss"], channel_std)
            loss.backward()
            optimizer.step()
        val_true, val_pred = predict(model, val_loader, device)
        val_mse = float(np.mean((val_pred.astype(np.float64) - val_true.astype(np.float64)) ** 2))
        scheduler.step(val_mse)
        if val_mse < best_val_mse - 1e-7:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("No valid checkpoint")
    model.load_state_dict(best_state)
    val_true, val_pred = predict(model, val_loader, device)
    val_result = BASE.metrics(val_true, val_pred)
    result = {f"val_{key}": value for key, value in val_result.items()}
    result.update(
        {
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "training_seconds": training_seconds,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "state_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        }
    )
    test_true = None
    test_pred = None
    if test_data is not None:
        x_test, y_test = BASE.make_windows(test_data, cfg.seq_len, cfg.pred_len)
        test_loader = DataLoader(
            BASE.WindowDataset(x_test, y_test), batch_size=max(batch_size, 512), shuffle=False
        )
        test_true, test_pred = predict(model, test_loader, device)
        result.update(BASE.metrics(test_true, test_pred))
    checkpoint = {
        "model_state": best_state,
        "variant": variant,
        "seed": seed,
        "cut_freq": cut_freq,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "channel_std": channel_std.cpu(),
    }
    return result, test_true, test_pred, checkpoint


def append(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def summarize(raw: pd.DataFrame, phase: str) -> pd.DataFrame:
    metric_names = ["val_RMSE", "val_MAE", "val_R2", "val_MAPE_pct", "val_sMAPE_pct"]
    if phase == "final":
        metric_names += ["RMSE", "MAE", "R2", "MAPE_pct", "sMAPE_pct"]
    records = []
    for (dataset, variant), group in raw.groupby(["dataset", "variant"], sort=False):
        record = {"dataset": dataset, "variant": variant, "runs": len(group)}
        for name in metric_names + ["best_epoch", "training_seconds", "parameter_count", "state_bytes"]:
            record[f"{name}_mean"] = float(group[name].mean())
            record[f"{name}_std"] = float(group[name].std(ddof=1)) if len(group) > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-parameter FITS optimization experiments")
    parser.add_argument("--phase", choices=["validation", "final"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=list(BASE.DATASETS), default=list(BASE.DATASETS))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027])
    parser.add_argument("--cut-freq", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--nrows", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "metrics_raw.csv"
    completed = set()
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        completed = set(old["run_id"])
    config = vars(args).copy()
    config["output"] = str(output)
    config["variant_definitions"] = VARIANTS
    config["selection_rule"] = "validation metrics only; test metrics are generated only in final phase"
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = BASE.ExperimentConfig(nrows=args.nrows)
    start = time.perf_counter()
    for dataset in args.datasets:
        data, meta = BASE.load_dataset(BASE.DATASETS[dataset], cfg)
        splits = BASE.split_series(data, cfg)
        dataset_dir = output / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset_metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for variant in args.variants:
            for seed in args.seeds:
                run_id = f"{dataset}|{variant}|{seed}"
                if run_id in completed:
                    print(f"SKIP {run_id}", flush=True)
                    continue
                print(f"RUN {run_id}", flush=True)
                result, y_true, y_pred, checkpoint = train_one(
                    splits["train"],
                    splits["val"],
                    splits["test"] if args.phase == "final" else None,
                    variant,
                    cfg,
                    seed,
                    args.cut_freq,
                    args.learning_rate,
                    args.batch_size,
                    args.max_epochs,
                    args.patience,
                )
                row = {"run_id": run_id, "dataset": dataset, "variant": variant, "seed": seed, **result}
                append(raw_path, row)
                torch.save(checkpoint, dataset_dir / f"{variant}_seed_{seed}.pt")
                if y_true is not None and y_pred is not None:
                    BASE.save_prediction(dataset_dir / f"{variant}_seed_{seed}_predictions.npz", y_true, y_pred)
                completed.add(run_id)
                displayed = result["RMSE"] if args.phase == "final" else result["val_RMSE"]
                print(
                    f"DONE {run_id} {args.phase}_RMSE={displayed:.6f} "
                    f"val_RMSE={result['val_RMSE']:.6f} epoch={result['best_epoch']}",
                    flush=True,
                )
                raw = pd.read_csv(raw_path)
                summarize(raw, args.phase).to_csv(output / "metrics_summary.csv", index=False)
    (output / "run_info.json").write_text(
        json.dumps({"status": "complete", "elapsed_seconds": time.perf_counter() - start}, indent=2),
        encoding="utf-8",
    )
    print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
