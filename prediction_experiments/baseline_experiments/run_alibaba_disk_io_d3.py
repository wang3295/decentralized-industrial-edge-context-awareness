from __future__ import annotations

"""Three-resource Alibaba experiment using disk I/O utilization.

This entry point reuses the exact model, calibration, budget, update-policy,
timing, and statistical code of ``run_alibaba_network_throughput_d3.py``.
Only the third resource and its data preparation are replaced: the state is
CPU utilization, memory utilization, and disk I/O utilization.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "run_alibaba_network_throughput_d3.py"
SPEC = importlib.util.spec_from_file_location("alibaba_d3_shared", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

CHANNELS = ("CPU", "Memory", "DiskIO")
BASE.CHANNELS = CHANNELS
BASE.ENTRY_SCRIPT = Path(__file__).resolve()


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
    npz_path = processed_dir / "alibaba_disk_io_d3.npz"
    manifest_path = processed_dir / "data_manifest.json"
    selected_csv = processed_dir / "alibaba_disk_io_d3_selected.csv"
    if npz_path.exists() and manifest_path.exists() and not force:
        saved = np.load(npz_path, allow_pickle=False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return saved["values"], saved["timestamps"], saved["machine_ids"], manifest

    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)
    rng = np.random.default_rng(seed)
    candidate_numbers = rng.choice(np.arange(1, 4001), size=100, replace=False)
    candidate_ids = [f"m_{number}" for number in candidate_numbers]
    candidate_set = set(candidate_ids)
    target_buffer = rows_per_machine + 500
    buffers: dict[str, list[pd.DataFrame]] = {machine_id: [] for machine_id in candidate_ids}
    counts = {machine_id: 0 for machine_id in candidate_ids}
    complete: set[str] = set()
    names = ["machine_id", "timestamp", "cpu", "memory", "disk_io"]

    print(
        f"Scanning complete {raw_csv} for {machine_count} seed-selected disk-I/O machines...",
        flush=True,
    )
    reader = pd.read_csv(
        raw_csv,
        header=None,
        usecols=[0, 1, 2, 3, 8],
        names=names,
        chunksize=1_000_000,
        low_memory=False,
    )
    chunks_read = 0
    invalid_disk_rows = 0
    observed_rows = 0
    for chunks_read, chunk in enumerate(reader, start=1):
        chunk = chunk.loc[chunk["machine_id"].isin(candidate_set)].copy()
        if chunk.empty:
            continue
        for column in names[1:]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        observed_rows += len(chunk)
        invalid_disk_rows += int(((chunk["disk_io"] < 0.0) | (chunk["disk_io"] > 100.0)).sum())
        chunk = chunk.dropna()
        valid = (
            chunk[["cpu", "memory", "disk_io"]].ge(0.0).all(axis=1)
            & chunk[["cpu", "memory", "disk_io"]].le(100.0).all(axis=1)
        )
        chunk = chunk.loc[valid]
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
            print(f"  chunks={chunks_read}, complete candidates={len(complete)}", flush=True)

    ordered_complete = [machine_id for machine_id in candidate_ids if machine_id in complete]
    if len(ordered_complete) < machine_count:
        raise RuntimeError(
            f"Only {len(ordered_complete)} candidates reached {target_buffer} valid rows"
        )
    selected_ids = ordered_complete[:machine_count]
    series: list[np.ndarray] = []
    timestamp_series: list[np.ndarray] = []
    export_frames: list[pd.DataFrame] = []
    for machine_id in selected_ids:
        frame = pd.concat(buffers[machine_id], ignore_index=True)
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if len(frame) < rows_per_machine:
            raise RuntimeError(f"{machine_id} has only {len(frame)} unique valid rows")
        frame = frame.iloc[:rows_per_machine].copy()
        normalized = frame[["cpu", "memory", "disk_io"]].to_numpy(dtype=np.float32) / 100.0
        series.append(BASE.causal_average(normalized, moving_average))
        timestamp_series.append(frame["timestamp"].to_numpy(dtype=np.int64))
        frame[["cpu", "memory", "disk_io"]] /= 100.0
        export_frames.append(frame)

    values = np.stack(series).astype(np.float32)
    timestamps = np.stack(timestamp_series)
    machine_ids = np.asarray(selected_ids)
    np.savez_compressed(npz_path, values=values, timestamps=timestamps, machine_ids=machine_ids)
    pd.concat(export_frames, ignore_index=True).to_csv(selected_csv, index=False)
    manifest: dict[str, object] = {
        "purpose": "proof-of-concept d=3 resource-state experiment using disk I/O",
        "raw_csv": str(raw_csv.resolve()),
        "raw_csv_size_bytes": raw_csv.stat().st_size,
        "official_archive_sha256": BASE.OFFICIAL_ARCHIVE_SHA256,
        "selection_seed": seed,
        "candidate_pool_size": len(candidate_ids),
        "chunks_read": chunks_read,
        "machine_count": machine_count,
        "rows_per_machine": rows_per_machine,
        "selected_machine_ids": selected_ids,
        "channels": list(CHANNELS),
        "disk_io_definition": "official disk_io_percent field divided by 100",
        "normalization": "CPU, memory, and disk I/O utilization divided by 100",
        "moving_average": f"causal rolling mean, window={moving_average}, applied per machine",
        "invalid_rule": "drop missing values and values outside [0,100] before selection",
        "candidate_rows_observed": observed_rows,
        "candidate_invalid_disk_rows": invalid_disk_rows,
        "npz_sha256": BASE.sha256_file(npz_path),
        "selected_csv_sha256": BASE.sha256_file(selected_csv),
        "values_shape": list(values.shape),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return values, timestamps, machine_ids, manifest


def write_report(
    output: Path,
    manifest: dict[str, object],
    prediction_summary: pd.DataFrame,
    update_summary: pd.DataFrame,
    selected: pd.DataFrame,
    environment: dict[str, object],
) -> None:
    pred = prediction_summary.set_index("model").loc[["FITS-Delta", "NaiveLastValue"]].reset_index()
    update = update_summary.set_index("method").loc[list(BASE.METHODS)].reset_index()
    lines = [
        "# Alibaba磁盘I/O第三资源维度实验结果与分析",
        "",
        "## 1. 目的与边界",
        "",
        "本实验检验FITS-Delta预测与状态更新机制能否由CPU/内存二维状态扩展到CPU、内存和磁盘I/O利用率三维状态。实验不构成三维候选发现或三维端到端卸载验证。",
        "",
        "## 2. 数据与协议",
        "",
        f"- 独立机器：{manifest['machine_count']}台；每台{manifest['rows_per_machine']}个有效观测。",
        f"- 机器选择种子：{manifest['selection_seed']}；完整扫描{manifest['chunks_read']}个百万行分块后选择机器。",
        "- 三维状态：归一化CPU利用率、内存利用率和官方`disk_io_percent`字段。",
        "- 每台机器内部按时间顺序70%/15%/15%划分，窗口不跨机器边界。",
        "- FITS-Delta输入100步、预测5步、cut frequency=24，运行5个独立种子；Naive为确定性对照。",
        "- 更新预算B=0.5，三维等权；所有方法每条轨迹的实际更新次数严格相同。",
        "- 所有超参数只在验证段选择。",
        f"- 环境：{environment.get('cpu', 'unknown CPU')}，PyTorch {environment['torch']}，{environment['torch_threads']}个推理线程。",
        "",
        "## 3. 三维预测结果",
        "",
        "| 模型 | Disk-I/O NRMSE↓ | Disk-I/O MAE↓ | Disk-I/O R²↑ | Macro NRMSE↓ | CPU RMSE↓ | Memory RMSE↓ | 参数量 | CPU中位/P95(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in pred.iterrows():
        lines.append(
            f"| {row.model} | {row.diskio_nrmse_mean:.4f} | {row.diskio_mae_mean:.4f} | "
            f"{row.diskio_r2_mean:.4f} | {row.macro_nrmse_mean:.4f} | {row.cpu_rmse_mean:.4f} | "
            f"{row.memory_rmse_mean:.4f} | {int(round(row.parameter_count_mean))} | "
            f"{row.latency_median_ms_mean:.4f}/{row.latency_p95_ms_mean:.4f} |"
        )
    lines.extend([
        "",
        "## 4. 三维状态更新结果",
        "",
        "| 更新策略 | 3D WRMSE↓ (95% CI) | Disk-I/O RMSE↓ | 平均AoI↓ | 更新率 | 决策中位/P95(ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in update.iterrows():
        lines.append(
            f"| {row.method} | {row.wrmse_d3_mean:.4f} ({row.wrmse_d3_ci95_low:.4f}–{row.wrmse_d3_ci95_high:.4f}) | "
            f"{row.diskio_rmse_mean:.4f} | {row.mean_aoi_mean:.3f} | {row.update_rate_mean:.4f} | "
            f"{row.decision_median_ms_mean:.5f}/{row.decision_p95_ms_mean:.5f} |"
        )
    winner = update.loc[update.wrmse_d3_mean.idxmin()]
    lines.extend([
        "",
        f"在相同更新预算下，三维WRMSE点估计最低的方法为{winner.method}（{winner.wrmse_d3_mean:.4f}）。",
        "",
        "## 5. 验证集参数",
        "",
        "| 种子 | Send-on-delta阈值 | Lyapunov V0 | MPC-DP eta |",
        "|---:|---:|---:|---:|",
    ])
    for _, row in selected.sort_values("seed").iterrows():
        lines.append(f"| {int(row.seed)} | {row.delta:.6g} | {row.lyapunov_v0:.6g} | {row.eta:.6g} |")
    lines.extend([
        "",
        "## 6. 结论边界",
        "",
        "本实验能够支持预测和状态更新代码容纳磁盘I/O这一额外资源维度。它不能证明网络带宽已被直接验证，也不能证明三维候选发现和完整卸载链路已经完成。",
        "",
    ])
    (output / "Alibaba磁盘IO第三资源维度实验结果与分析.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    BASE.prepare_data = prepare_data
    BASE.write_report = write_report
    if "--processed-dir" not in sys.argv:
        sys.argv.extend([
            "--processed-dir",
            str(HERE / "data" / "processed" / "alibaba_disk_io_d3"),
        ])
    if "--output-dir" not in sys.argv:
        sys.argv.extend([
            "--output-dir",
            str(HERE / "results" / "alibaba_disk_io_d3_formal_20260915"),
        ])
    BASE.main()


if __name__ == "__main__":
    main()
