"""Validate and aggregate five industrial-PC disk-I/O d=3 timing runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("Periodic", "Send-on-delta", "Lyapunov", "MPC-DP")
NON_TIMING = (
    "wrmse_d3_mean",
    "cpu_rmse_mean",
    "memory_rmse_mean",
    "diskio_rmse_mean",
    "update_count_mean",
    "update_rate_mean",
    "mean_aoi_mean",
    "p95_aoi_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--industrial-dir", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--processed-npz", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_runs(root: Path) -> pd.DataFrame:
    directories = sorted(
        (path for path in root.glob("formal_run_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    if len(directories) != 5:
        raise ValueError(f"Expected five process runs, found {len(directories)}")
    frames = []
    for run_index, directory in enumerate(directories, start=1):
        frame = pd.read_csv(directory / "update_summary.csv")
        if tuple(frame["method"]) != METHODS:
            raise ValueError(f"Unexpected method order in {directory}")
        frame["process_run"] = run_index
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def verify_hashes(root: Path, formal: Path, processed_npz: Path) -> pd.DataFrame:
    metadata = json.loads((root / "formal_run_1" / "run_metadata.json").read_text(encoding="utf-8"))
    local_paths = {
        "alibaba_disk_io_d3.npz": processed_npz,
        "selected_hyperparameters.csv": formal / "selected_hyperparameters.csv",
    }
    for seed in (2026, 2027, 2028, 2029, 2030):
        name = f"fits_delta_d3_seed_{seed}_predictions.npz"
        local_paths[name] = formal / name
    rows = []
    for name, path in local_paths.items():
        local_hash = sha256(path)
        industrial_hash = metadata["input_sha256"][name]
        rows.append(
            {
                "file": name,
                "local_sha256": local_hash,
                "industrial_sha256": industrial_hash,
                "match": local_hash == industrial_hash,
            }
        )
    result = pd.DataFrame(rows)
    if not result["match"].all():
        raise ValueError("Industrial inputs do not match the formal local inputs")
    return result


def verify_metrics(runs: pd.DataFrame, formal: Path) -> pd.DataFrame:
    reference = pd.read_csv(formal / "update_summary.csv").set_index("method")
    rows = []
    for method in METHODS:
        group = runs[runs["method"] == method]
        record = {"method": method}
        max_difference = 0.0
        for column in NON_TIMING:
            reference_value = float(reference.loc[method, column])
            differences = np.abs(group[column].to_numpy(dtype=float) - reference_value)
            max_difference = max(max_difference, float(np.max(differences)))
        record["max_abs_difference_vs_formal"] = max_difference
        record["match_within_1e-9"] = max_difference <= 1e-9
        rows.append(record)
    result = pd.DataFrame(rows)
    if not result["match_within_1e-9"].all():
        raise ValueError("Industrial non-timing metrics differ from the formal experiment")
    return result


def aggregate_timing(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        group = runs[runs["method"] == method]
        medians = group["decision_median_ms_mean"].to_numpy(dtype=float)
        p95s = group["decision_p95_ms_mean"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "independent_process_runs": len(group),
                "industrial_median_decision_ms": float(np.median(medians)),
                "industrial_median_decision_ms_min": float(np.min(medians)),
                "industrial_median_decision_ms_max": float(np.max(medians)),
                "industrial_p95_decision_ms": float(np.median(p95s)),
                "industrial_p95_decision_ms_min": float(np.min(p95s)),
                "industrial_p95_decision_ms_max": float(np.max(p95s)),
            }
        )
    return pd.DataFrame(rows)


def build_table(formal: Path, timing: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(formal / "update_summary.csv").set_index("method")
    timing = timing.set_index("method")
    rows = []
    for method in METHODS:
        rows.append(
            {
                "method": method,
                "wrmse_d3": reference.loc[method, "wrmse_d3_mean"],
                "wrmse_d3_ci95_low": reference.loc[method, "wrmse_d3_ci95_low"],
                "wrmse_d3_ci95_high": reference.loc[method, "wrmse_d3_ci95_high"],
                "diskio_rmse": reference.loc[method, "diskio_rmse_mean"],
                "mean_aoi": reference.loc[method, "mean_aoi_mean"],
                "update_rate": reference.loc[method, "update_rate_mean"],
                "industrial_median_decision_ms": timing.loc[
                    method, "industrial_median_decision_ms"
                ],
                "industrial_p95_decision_ms": timing.loc[
                    method, "industrial_p95_decision_ms"
                ],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.industrial_dir)
    hashes = verify_hashes(args.industrial_dir, args.formal_dir, args.processed_npz)
    consistency = verify_metrics(runs, args.formal_dir)
    timing = aggregate_timing(runs)
    table = build_table(args.formal_dir, timing)

    runs.to_csv(args.industrial_dir / "five_process_timing_runs.csv", index=False)
    hashes.to_csv(args.industrial_dir / "input_hash_check.csv", index=False)
    consistency.to_csv(args.industrial_dir / "non_timing_consistency_check.csv", index=False)
    timing.to_csv(args.industrial_dir / "industrial_timing_aggregate.csv", index=False)
    table.to_csv(args.industrial_dir / "manuscript_table.csv", index=False)

    lines = [
        "# Alibaba磁盘I/O三维状态更新：工控机决策时延复测",
        "",
        "- 平台：TD-iTX-17121U，Intel Core i5-1235U，Linux。",
        "- Python 3.12.14，NumPy 2.0.1，单核固定，五个独立Python进程。",
        "- 输入为正式实验冻结的测试轨迹、五组FITS-Delta预测和验证集参数；未重新训练或调参。",
        "- 表中时延为五个独立进程中相应汇总统计量的中位数。",
        "- 所有输入SHA-256一致；非计时指标与正式实验的最大绝对差不超过1e-9。",
        "",
        "| 更新策略 | 3D WRMSE↓（95% CI） | Disk-I/O RMSE↓ | 平均AoI↓ | 更新率 | 工控机决策中位/P95（ms）↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['wrmse_d3']:.5f} "
            f"({row['wrmse_d3_ci95_low']:.5f}–{row['wrmse_d3_ci95_high']:.5f}) | "
            f"{row['diskio_rmse']:.5f} | {row['mean_aoi']:.3f} | "
            f"{row['update_rate']:.3f} | {row['industrial_median_decision_ms']:.5f} / "
            f"{row['industrial_p95_decision_ms']:.5f} |"
        )
    (args.industrial_dir / "工控机磁盘IO三维更新时延复测.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
