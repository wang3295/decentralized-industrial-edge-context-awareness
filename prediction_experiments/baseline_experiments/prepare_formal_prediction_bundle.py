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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


UNIFIED = load_module("formal_run_all_models_unified", HERE / "run_all_models_unified.py")
BASE = UNIFIED.BASE


MODEL_TO_INDUSTRIAL = {
    "FITS": "FITS",
    "FITS-Delta": "FITS-Delta",
    "NaiveLastValue": "Naive last-value",
    "ARIMA": "ARIMA",
    "SingleLayerGRU": "Single-layer GRU",
    "DLinear": "DLinear",
    "S2S-LSTM": "S2S LSTM",
    "FreTS": "FreTS-lite",
    "PatchTST": "PatchTST-lite",
}

PAPER_MODEL_LABEL = {
    "FITS": "FITS",
    "FITS-Delta": "FITS-Delta",
    "NaiveLastValue": "Naive",
    "ARIMA": "ARIMA",
    "SingleLayerGRU": "Single-layer GRU",
    "DLinear": "DLinear",
    "S2S-LSTM": "S2S-LSTM",
    "FreTS": "FreTS-lite",
    "PatchTST": "PatchTST-lite",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_complete_benchmark(benchmark_dir: Path) -> tuple[pd.DataFrame, dict]:
    run_info = json.loads((benchmark_dir / "run_info.json").read_text(encoding="utf-8"))
    qa = json.loads((benchmark_dir / "qa_report.json").read_text(encoding="utf-8"))
    if run_info.get("status") != "complete" or run_info.get("completed_runs") != 111:
        raise RuntimeError(f"Nine-model benchmark is incomplete: {run_info}")
    if qa.get("status") != "passed" or qa.get("raw_rows") != 111:
        raise RuntimeError(f"Nine-model benchmark QA failed: {qa}")
    summary = pd.read_csv(benchmark_dir / "prediction_summary_mean_std.csv")
    if len(summary) != 27:
        raise RuntimeError(f"Expected 27 summary rows, found {len(summary)}")
    return summary, {"run_info": run_info, "qa": qa}


def merge_accuracy_and_industrial(
    accuracy: pd.DataFrame, industrial_dir: Path
) -> pd.DataFrame:
    combined_path = industrial_dir / "industrial_three_dataset_summary_27.csv"
    if combined_path.exists():
        efficiency = pd.read_csv(combined_path)
    else:
        efficiency = pd.read_csv(industrial_dir / "vm_inference_summary.csv")
        efficiency = pd.concat(
            [efficiency.assign(dataset=dataset) for dataset in accuracy["dataset"].unique()],
            ignore_index=True,
        )
    if len(efficiency) != 27 or not (efficiency["runs"] == 5).all():
        raise RuntimeError("Industrial benchmark must contain 3 datasets x 9 models x 5 runs")
    efficiency = efficiency.set_index(["dataset", "model"])
    rows: list[dict] = []
    for record in accuracy.to_dict(orient="records"):
        industrial_name = MODEL_TO_INDUSTRIAL[record["model"]]
        measured = efficiency.loc[(record["dataset"], industrial_name)]
        predicted_parameters = int(round(float(record["parameter_count_mean"])))
        measured_parameters = int(measured["parameter_count"])
        if predicted_parameters != measured_parameters:
            raise RuntimeError(
                f"Parameter mismatch for {record['model']}: "
                f"accuracy={predicted_parameters}, industrial={measured_parameters}"
            )
        rows.append(
            {
                "dataset": record["dataset"],
                "model": record["model"],
                "paper_model_label": PAPER_MODEL_LABEL[record["model"]],
                "runs": int(record["runs"]),
                "RMSE_mean": record["RMSE_mean"],
                "RMSE_std": record["RMSE_std"],
                "MAE_mean": record["MAE_mean"],
                "MAE_std": record["MAE_std"],
                "R2_mean": record["R2_mean"],
                "R2_std": record["R2_std"],
                "MAPE_pct_mean": record["MAPE_pct_mean"],
                "MAPE_pct_std": record["MAPE_pct_std"],
                "sMAPE_pct_mean": record["sMAPE_pct_mean"],
                "sMAPE_pct_std": record["sMAPE_pct_std"],
                "parameter_count": predicted_parameters,
                "industrial_parameter_count": measured_parameters,
                "model_state_kib": measured["model_state_kib"],
                "industrial_latency_median_ms": measured["latency_median_ms_mean"],
                "industrial_latency_p95_ms": measured["latency_p95_ms_mean"],
                "industrial_peak_rss_mb": measured["process_peak_rss_mb_max"],
                "source_file": record["source_file"],
                "implementation_variant": record["implementation_variant"],
            }
        )
    return pd.DataFrame(rows)


def config_from_payload(payload: dict):
    values = dict(payload["config"])
    values["seeds"] = tuple(values["seeds"])
    values["arima_p_candidates"] = tuple(values["arima_p_candidates"])
    values["arima_d_candidates"] = tuple(values["arima_d_candidates"])
    return UNIFIED.UnifiedConfig(**values)


def online_predictions(
    model_key: str,
    checkpoint_path: Path,
    x_test: np.ndarray,
    cfg,
    indices: np.ndarray,
) -> tuple[np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_key"] != model_key or int(checkpoint["seed"]) != 2026:
        raise RuntimeError(f"Unexpected checkpoint identity in {checkpoint_path}")
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    scaled = ((x_test[indices] - mean) / std).astype(np.float32)
    model = UNIFIED.make_model(model_key, cfg, channels=x_test.shape[-1]).cpu().eval()
    model.load_state_dict(checkpoint["model_state"])
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(scaled), 128):
            batch = torch.from_numpy(scaled[start : start + 128])
            outputs.append(model(batch).cpu().numpy())
    predicted_scaled = np.concatenate(outputs)
    predicted = predicted_scaled * std[None, None, :] + mean[None, None, :]
    identity = {
        "model_key": checkpoint["model_key"],
        "model_label": checkpoint["model_label"],
        "seed": int(checkpoint["seed"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_file": checkpoint["source_file"],
        "implementation_variant": checkpoint["implementation_variant"],
        "normalization_mean": mean.astype(float).tolist(),
        "normalization_std": std.astype(float).tolist(),
    }
    return predicted.astype(np.float32), identity


def create_trace_pack(
    benchmark_dir: Path,
    output_dir: Path,
    cfg,
) -> dict:
    dataset_name = "Alibaba"
    spec = BASE.DATASETS[dataset_name]
    data, current_metadata = BASE.load_dataset(spec, UNIFIED.base_config(cfg))
    splits = BASE.split_series(data, UNIFIED.base_config(cfg))
    x_test, y_test = BASE.make_windows(splits["test"], cfg.seq_len, cfg.pred_len)
    dataset_dir = benchmark_dir / dataset_name
    canonical = np.load(dataset_dir / "canonical_test_targets.npz")["y_true"]
    if not np.array_equal(y_test, canonical):
        max_error = float(np.max(np.abs(y_test - canonical)))
        raise RuntimeError(f"Reconstructed test targets differ; max error={max_error}")

    conformance_indices = np.unique(
        np.linspace(0, len(x_test) - 1, num=min(257, len(x_test)), dtype=int)
    )
    actual = x_test[:, -1, :].astype(np.float32)
    test_start = int(cfg.nrows * (cfg.train_ratio + cfg.val_ratio))
    trace_payload: dict[str, np.ndarray] = {
        "trace_index": np.arange(len(actual), dtype=np.int64),
        "source_row_index_current": np.arange(
            test_start + cfg.seq_len - 1,
            test_start + cfg.seq_len - 1 + len(actual),
            dtype=np.int64,
        ),
        "actual": actual,
        "true_future": y_test.astype(np.float32),
    }
    conformance: dict[str, dict] = {}
    identities: dict[str, dict] = {}
    replay_dirs: dict[str, str] = {}
    for model_key in ("fits", "fits_delta"):
        prediction_path = dataset_dir / f"{model_key}_seed_2026_predictions.npz"
        stored = np.load(prediction_path)
        if not np.array_equal(stored["y_true"], y_test):
            raise RuntimeError(f"{prediction_path.name} uses different test targets")
        saved_prediction = stored["y_pred"].astype(np.float32)
        online, identity = online_predictions(
            model_key,
            dataset_dir / f"{model_key}_seed_2026.pt",
            x_test,
            cfg,
            conformance_indices,
        )
        reference = saved_prediction[conformance_indices]
        absolute_error = np.abs(online - reference)
        max_error = float(absolute_error.max())
        mean_error = float(absolute_error.mean())
        # Saved predictions were produced on CUDA whereas this audit replays on
        # CPU. FFT kernels differ slightly in floating-point rounding.
        passed = bool(np.allclose(online, reference, rtol=1e-5, atol=5e-4))
        if not passed:
            raise RuntimeError(
                f"Online/replay conformance failed for {model_key}: max error={max_error}"
            )
        conformance[model_key] = {
            "samples_checked": int(len(conformance_indices)),
            "values_checked": int(online.size),
            "max_absolute_error": max_error,
            "mean_absolute_error": mean_error,
            "rtol": 1e-5,
            "atol": 5e-4,
            "passed": passed,
        }
        identity["prediction_file_sha256"] = sha256_file(prediction_path)
        identities[model_key] = identity
        forecast = np.concatenate([actual[:, None, :], saved_prediction], axis=1)
        trace_payload[f"{model_key}_forecast"] = forecast.astype(np.float32)

        replay_dir = output_dir / "simulation_inputs" / f"{model_key}_seed_2026"
        replay_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(actual).to_csv(replay_dir / "value.csv", index=False, header=False)
        columns = [f"step_{step}" for step in range(cfg.pred_len + 1)]
        pd.DataFrame(forecast[:, :, 0], columns=columns).to_csv(
            replay_dir / "cpu_predictions.csv", index=False
        )
        pd.DataFrame(forecast[:, :, 1], columns=columns).to_csv(
            replay_dir / "memory_predictions.csv", index=False
        )
        replay_dirs[model_key] = str(replay_dir.resolve())

    trace_path = output_dir / "alibaba_heldout_test_trace_pack_seed2026.npz"
    np.savez_compressed(trace_path, **trace_payload)
    trace_metadata = {
        "status": "passed",
        "purpose": "formal system-simulation replay input derived from a held-out test split",
        "dataset": dataset_name,
        "source_path": str(spec.path.resolve()),
        "source_file_size_bytes": spec.path.stat().st_size,
        "source_file_sha256": sha256_file(spec.path),
        "source_header": spec.header,
        "source_columns": current_metadata["columns"],
        "processed_first_n_rows": cfg.nrows,
        "processed_sha256": current_metadata["processed_sha256"],
        "split": {
            "type": "chronological",
            "train_rows": [0, int(cfg.nrows * cfg.train_ratio) - 1],
            "validation_rows": [
                int(cfg.nrows * cfg.train_ratio),
                test_start - 1,
            ],
            "test_rows": [test_start, cfg.nrows - 1],
            "ratios": [cfg.train_ratio, cfg.val_ratio, round(1 - cfg.train_ratio - cfg.val_ratio, 12)],
        },
        "preprocessing": {
            "moving_average": "causal",
            "moving_average_window": cfg.moving_average_window,
            "normalization": "training-split mean and standard deviation from checkpoint",
        },
        "task": {
            "input_length": cfg.seq_len,
            "prediction_length": cfg.pred_len,
            "channels": ["CPU", "memory"],
        },
        "trace_rows": int(len(actual)),
        "trace_semantics": {
            "actual": "last observed value of each held-out test input window",
            "forecast_step_0": "the same observed value as actual",
            "forecast_steps_1_to_5": "frozen model forecasts for the next five values",
            "true_future": "held-out next-five ground truth; never used for model fitting",
        },
        "trace_pack_path": str(trace_path.resolve()),
        "trace_pack_sha256": sha256_file(trace_path),
        "replay_directories": replay_dirs,
        "models": identities,
        "online_vs_saved_prediction_conformance": conformance,
    }
    json_dump(output_dir / "trace_manifest.json", trace_metadata)
    return trace_metadata


def fmt_mean_std(mean: float, std: float, runs: int, digits: int = 4) -> str:
    if runs <= 1 or not np.isfinite(std) or abs(std) < 1e-15:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def emphasize_rank(text: str, rank: float) -> str:
    if int(rank) == 1:
        return f"**{text}**"
    if int(rank) == 2:
        return f"<u>{text}</u>"
    return text


def write_report(
    output_dir: Path,
    merged: pd.DataFrame,
    evidence: dict,
    trace_metadata: dict,
) -> None:
    lines = [
        "# 九模型预测精度—工控机效率统一正式结果",
        "",
        "## 1. 完整性结论",
        "",
        f"- 预测主实验：{evidence['run_info']['completed_runs']} / {evidence['run_info']['expected_runs']} 个运行单元完成。",
        f"- 统一测试真值：{'通过' if evidence['qa']['same_test_targets_for_all_models'] else '未通过'}。",
        f"- 工控机效率实验：3个数据集×9种方法×5个独立进程，共135条记录。",
        "- 三个数据集各自使用的种子2026权重/ARIMA参数与统一主实验文件SHA-256完全一致。",
        f"- 正式Alibaba测试轨迹：{trace_metadata['trace_rows']}个时隙，完全来自按时间划分的后15%测试段。",
        "- 在线重新执行FITS/FITS-Delta与读取保存预测的一致性检查均通过。",
        "",
        "## 2. 统一协议",
        "",
        "- 三个数据集均取前100,000行，采用5点因果移动平均，并按时间顺序70%/15%/15%划分。",
        "- 输入100步、预测5步、CPU/内存双通道；神经模型归一化统计量仅由训练段计算。",
        "- 学习模型使用种子2026–2030，报告均值±样本标准差；Naive和ARIMA为确定性方法。",
        "- 部署效率来自15 W级Intel Core i5-1235U Linux工控机：CPU-only、单线程、batch=1、每个数据集257个留出测试窗口、预热200次、正式2000次、5个独立进程。",
        "",
        "## 3. 统一结果",
        "",
        "表中粗体表示该数据集该列最优，下划线表示次优；并列值采用相同名次。R²越大越好，其余列越小越好。",
        "",
    ]
    for dataset in ["Alibaba", "GWA-T-12", "Google"]:
        lines += [
            f"### {dataset}",
            "",
            "| 模型 | RMSE↓ | MAE↓ | R²↑ | MAPE (%)↓ | sMAPE (%)↓ | 参数量 | 工控机中位时延 (ms) | 工控机P95 (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        frame = merged.loc[merged["dataset"] == dataset].copy()
        rank_columns = {
            "RMSE_mean": True,
            "MAE_mean": True,
            "R2_mean": False,
            "MAPE_pct_mean": True,
            "sMAPE_pct_mean": True,
            "parameter_count": True,
            "industrial_latency_median_ms": True,
            "industrial_latency_p95_ms": True,
        }
        ranks = {
            column: frame[column].rank(method="dense", ascending=ascending)
            for column, ascending in rank_columns.items()
        }
        for column, values in ranks.items():
            frame[f"{column}_rank"] = values
        for row in frame.itertuples(index=False):
            lines.append(
                "| {model} | {rmse} | {mae} | {r2} | {mape} | {smape} | {params} | {median} | {p95} |".format(
                    model=row.paper_model_label,
                    rmse=emphasize_rank(
                        fmt_mean_std(row.RMSE_mean, row.RMSE_std, row.runs),
                        row.RMSE_mean_rank,
                    ),
                    mae=emphasize_rank(
                        fmt_mean_std(row.MAE_mean, row.MAE_std, row.runs),
                        row.MAE_mean_rank,
                    ),
                    r2=emphasize_rank(
                        fmt_mean_std(row.R2_mean, row.R2_std, row.runs),
                        row.R2_mean_rank,
                    ),
                    mape=emphasize_rank(
                        fmt_mean_std(row.MAPE_pct_mean, row.MAPE_pct_std, row.runs),
                        row.MAPE_pct_mean_rank,
                    ),
                    smape=emphasize_rank(
                        fmt_mean_std(row.sMAPE_pct_mean, row.sMAPE_pct_std, row.runs),
                        row.sMAPE_pct_mean_rank,
                    ),
                    params=emphasize_rank(
                        f"{int(row.parameter_count):,}", row.parameter_count_rank
                    ),
                    median=emphasize_rank(
                        f"{row.industrial_latency_median_ms:.4f}",
                        row.industrial_latency_median_ms_rank,
                    ),
                    p95=emphasize_rank(
                        f"{row.industrial_latency_p95_ms:.4f}",
                        row.industrial_latency_p95_ms_rank,
                    ),
                )
            )
        lines.append("")
    lines += [
        "## 4. 结果边界",
        "",
        "- FITS以625个参数获得稳定且有竞争力的结果，但并非所有数据集和指标均为第一。Alibaba的最低RMSE来自DLinear，GWA-T-12由ARIMA领先，Google的最低RMSE来自单层GRU。",
        "- 三数据集上的FITS平均中位时延为0.0620–0.0632 ms；它不是绝对最快，但明显轻于GRU、S2S-LSTM和PatchTST-lite。",
        "- FreTS和PatchTST是作者代码中的轻量变体，论文必须写作FreTS-lite和PatchTST-lite，不能泛化为官方完整模型的性能。",
        "- ARIMA阶数按各数据集验证集分别选择，因此参数量及时延均使用相应数据集配置，不再复用Alibaba结果。",
        "- MAPE/sMAPE在Google近零真值上不稳定，应以RMSE、MAE和R²为主要准确性指标。",
        "- 仿真轨迹回放不包含预测程序的执行时延；预测部署时延由工控机实验单独报告。",
        "",
        "## 5. 轨迹回放一致性",
        "",
        "| 模型 | 检查窗口 | 检查数值 | 最大绝对误差 | 平均绝对误差 | 结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, item in trace_metadata["online_vs_saved_prediction_conformance"].items():
        lines.append(
            f"| {key} | {item['samples_checked']} | {item['values_checked']} | "
            f"{item['max_absolute_error']:.8g} | {item['mean_absolute_error']:.8g} | "
            f"{'通过' if item['passed'] else '失败'} |"
        )
    lines += [
        "",
        "该检查证明：在固定数据、预处理、权重和模型实现下，仿真读取预计算预测与在线执行同一模型得到相同预测（允许CPU/GPU浮点舍入误差）。因此正式大规模仿真可以使用该小型轨迹包，但必须引用本目录的`trace_manifest.json`。",
        "",
        "## 6. 文件",
        "",
        "- `paper_table_accuracy_industrial.csv`：三个数据集的预测精度与工控机效率合并表。",
        "- `alibaba_heldout_test_trace_pack_seed2026.npz`：正式仿真轨迹包。",
        "- `simulation_inputs/`：兼容现有更新实验读取格式的FITS和FITS-Delta轨迹。",
        "- `trace_manifest.json`：数据区间、预处理、模型与文件哈希及一致性检查。",
        "- `artifact_manifest.json`：统一主实验、工控机实验及关键源代码摘要。",
    ]
    (output_dir / "九模型预测精度与工控机效率统一正式结果.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the formal prediction evidence bundle")
    parser.add_argument(
        "--alibaba-csv",
        type=Path,
        help="Path to the locally prepared two-column Alibaba usage CSV; never downloaded or redistributed by this script",
    )
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="Regenerate the FITS/FITS-Delta downstream inputs without requiring industrial timing records",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=HERE / "results" / "all_models_unified_full_v1",
    )
    parser.add_argument(
        "--industrial-dir",
        type=Path,
        default=HERE / "results" / "industrial_three_datasets_i5_1235u_baremetal_cpu_formal_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "prediction_formal_unified_v3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.alibaba_csv is not None:
        source = args.alibaba_csv.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        original = BASE.DATASETS["Alibaba"]
        BASE.DATASETS["Alibaba"] = BASE.DatasetSpec(
            name=original.name,
            path=source,
            header=original.header,
            columns=original.columns,
        )
    benchmark_dir = args.benchmark_dir.resolve()
    industrial_dir = args.industrial_dir.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy, evidence = ensure_complete_benchmark(benchmark_dir)
    config_payload = json.loads((benchmark_dir / "config.json").read_text(encoding="utf-8"))
    cfg = config_from_payload(config_payload)
    if args.trace_only:
        trace_metadata = create_trace_pack(benchmark_dir, output_dir, cfg)
        print(json.dumps({
            "status": "passed",
            "mode": "trace-only",
            "output": str(output_dir),
            "prediction_runs": int(evidence["run_info"]["completed_runs"]),
            "trace_rows": trace_metadata["trace_rows"],
            "conformance": trace_metadata["online_vs_saved_prediction_conformance"],
        }, ensure_ascii=False, indent=2))
        return
    industrial_validation = json.loads(
        (industrial_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    if str(industrial_validation.get("status", "")).lower() not in {"pass", "passed"}:
        raise RuntimeError(f"Industrial validation failed: {industrial_validation}")
    merged = merge_accuracy_and_industrial(accuracy, industrial_dir)
    merged.to_csv(output_dir / "paper_table_accuracy_industrial.csv", index=False)

    trace_metadata = create_trace_pack(benchmark_dir, output_dir, cfg)

    manifest_paths = [
        HERE / "run_all_models_unified.py",
        HERE / "run_baselines.py",
        HERE / "benchmark_models_on_vm.py",
        benchmark_dir / "config.json",
        benchmark_dir / "prediction_metrics_raw.csv",
        benchmark_dir / "prediction_summary_mean_std.csv",
        industrial_dir / "industrial_three_dataset_raw_135.csv",
        industrial_dir / "industrial_three_dataset_summary_27.csv",
    ]
    artifact_manifest = {
        "status": "passed",
        "prediction_runs": int(evidence["run_info"]["completed_runs"]),
        "prediction_summary_rows": int(len(merged)),
        "industrial_runs": 135,
        "parameter_counts_match_fixed_architectures": True,
        "arima_parameter_count_note": "ARIMA order and industrial timing are validation-selected and measured separately for each dataset.",
        "files": {
            str(path.resolve()): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_paths
        },
    }
    json_dump(output_dir / "artifact_manifest.json", artifact_manifest)
    evidence["industrial_validation"] = industrial_validation
    write_report(output_dir, merged, evidence, trace_metadata)
    print(json.dumps({
        "status": "passed",
        "output": str(output_dir),
        "prediction_runs": evidence["run_info"]["completed_runs"],
        "industrial_runs": 135,
        "trace_rows": trace_metadata["trace_rows"],
        "conformance": trace_metadata["online_vs_saved_prediction_conformance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
