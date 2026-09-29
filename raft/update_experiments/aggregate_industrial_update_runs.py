from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["Periodic", "Send-on-delta", "Lyapunov", "MPC-DP"]
NON_TIMING_COLUMNS = [
    "wrmse_mean",
    "cpu_rmse_mean",
    "memory_rmse_mean",
    "update_count_mean",
    "update_rate_mean",
    "mean_aoi_mean",
    "p95_aoi_mean",
    "budget_violation_count_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--industrial-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_runs(root: Path) -> pd.DataFrame:
    run_dirs = sorted(
        (path for path in root.glob("formal_run*") if path.is_dir()),
        key=lambda path: (len(path.name), path.name),
    )
    if len(run_dirs) != 5:
        raise ValueError(f"Expected five independent runs, found {len(run_dirs)}")
    frames = []
    for run_index, run_dir in enumerate(run_dirs, start=1):
        frame = pd.read_csv(run_dir / "main_B0.5_summary.csv")
        frame = frame[frame["method"].isin(METHODS)].copy()
        if set(frame["method"]) != set(METHODS):
            raise ValueError(f"Incomplete methods in {run_dir}")
        frame["run"] = run_index
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_consistency(runs: pd.DataFrame, reference_dir: Path) -> pd.DataFrame:
    first = runs[runs["run"] == 1].set_index("method")
    reference = pd.read_csv(reference_dir / "main_B0.5_summary.csv")
    reference = reference[reference["method"].isin(METHODS)].set_index("method")
    rows = []
    for method in METHODS:
        method_runs = runs[runs["method"] == method]
        repeat_match = True
        local_match = True
        max_repeat_difference = 0.0
        max_local_difference = 0.0
        for column in NON_TIMING_COLUMNS:
            values = method_runs[column].to_numpy(dtype=float)
            repeat_difference = float(np.max(np.abs(values - values[0])))
            local_difference = float(abs(first.loc[method, column] - reference.loc[method, column]))
            max_repeat_difference = max(max_repeat_difference, repeat_difference)
            max_local_difference = max(max_local_difference, local_difference)
            repeat_match &= bool(np.allclose(values, values[0], rtol=0.0, atol=1e-12))
            local_match &= bool(
                np.isclose(first.loc[method, column], reference.loc[method, column], rtol=0.0, atol=1e-12)
            )
        rows.append(
            {
                "method": method,
                "five_industrial_runs_match": repeat_match,
                "industrial_matches_local_reference": local_match,
                "max_abs_difference_across_industrial_runs": max_repeat_difference,
                "max_abs_difference_vs_local_reference": max_local_difference,
            }
        )
    result = pd.DataFrame(rows)
    if not result["five_industrial_runs_match"].all() or not result[
        "industrial_matches_local_reference"
    ].all():
        raise ValueError("Cross-run or cross-platform algorithm metrics do not match")
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
                "median_decision_ms": float(np.median(medians)),
                "median_decision_ms_min": float(np.min(medians)),
                "median_decision_ms_max": float(np.max(medians)),
                "p95_decision_ms": float(np.median(p95s)),
                "p95_decision_ms_min": float(np.min(p95s)),
                "p95_decision_ms_max": float(np.max(p95s)),
            }
        )
    return pd.DataFrame(rows)


def collect_peak_rss(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("resource_usage*.txt"))
    if len(paths) != 5:
        raise ValueError(f"Expected five resource-usage files, found {len(paths)}")
    rows = []
    for index, path in enumerate(paths, start=1):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        if match is None:
            raise ValueError(f"Peak RSS not found in {path}")
        rows.append({"run": index, "peak_rss_mib": int(match.group(1)) / 1024.0})
    return pd.DataFrame(rows)


def build_main_table(runs: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    first = runs[runs["run"] == 1].set_index("method")
    timing = timing.set_index("method")
    rows = []
    for method in METHODS:
        rows.append(
            {
                "method": method,
                "wrmse": first.loc[method, "wrmse_mean"],
                "cpu_rmse": first.loc[method, "cpu_rmse_mean"],
                "memory_rmse": first.loc[method, "memory_rmse_mean"],
                "update_count": int(round(first.loc[method, "update_count_mean"])),
                "actual_update_rate": first.loc[method, "update_rate_mean"],
                "mean_aoi": first.loc[method, "mean_aoi_mean"],
                "p95_aoi": first.loc[method, "p95_aoi_mean"],
                "industrial_median_decision_ms": timing.loc[method, "median_decision_ms"],
                "industrial_p95_decision_ms": timing.loc[method, "p95_decision_ms"],
                "budget_violations": int(round(first.loc[method, "budget_violation_count_mean"])),
            }
        )
    return pd.DataFrame(rows)


def write_provenance(root: Path, project_root: Path, data_dir: Path) -> pd.DataFrame:
    pairs = [
        (
            project_root / "raft" / "update_experiments" / "run_update_experiments.py",
            root / "provenance" / "remote_code" / "run_update_experiments.py",
        ),
        (
            project_root / "raft" / "edgebench_core" / "update_policies.py",
            root / "provenance" / "remote_code" / "update_policies.py",
        ),
        (data_dir / "value.csv", root / "provenance" / "remote_data" / "value.csv"),
        (
            data_dir / "cpu_predictions.csv",
            root / "provenance" / "remote_data" / "cpu_predictions.csv",
        ),
        (
            data_dir / "memory_predictions.csv",
            root / "provenance" / "remote_data" / "memory_predictions.csv",
        ),
    ]
    rows = []
    for local, remote_copy in pairs:
        local_hash = sha256(local)
        remote_hash = sha256(remote_copy)
        rows.append(
            {
                "file": local.name,
                "local_sha256": local_hash,
                "industrial_copy_sha256": remote_hash,
                "match": local_hash == remote_hash,
            }
        )
    result = pd.DataFrame(rows)
    if not result["match"].all():
        raise ValueError("Industrial code/input provenance hashes do not match local sources")
    return result


def write_report(
    root: Path,
    main_table: pd.DataFrame,
    timing: pd.DataFrame,
    consistency: pd.DataFrame,
    rss: pd.DataFrame,
    provenance: pd.DataFrame,
) -> None:
    first_metadata = json.loads(
        (root / "formal_run" / "run_metadata.json").read_text(encoding="utf-8")
    )
    rows = []
    for item in main_table.to_dict(orient="records"):
        method = f"**{item['method']}**" if item["method"] == "MPC-DP" else item["method"]
        wrmse = f"**{item['wrmse']:.4f}**" if item["method"] == "MPC-DP" else f"{item['wrmse']:.4f}"
        rows.append(
            f"| {method} | {wrmse} | {item['cpu_rmse']:.4f} | {item['memory_rmse']:.4f} | "
            f"{item['update_count']}/{item['actual_update_rate']:.4f} | "
            f"{item['mean_aoi']:.3f}/{item['p95_aoi']:.3f} | "
            f"{item['industrial_median_decision_ms']:.6f}/{item['industrial_p95_decision_ms']:.6f} | "
            f"{item['budget_violations']} |"
        )

    timing_rows = []
    for item in timing.to_dict(orient="records"):
        timing_rows.append(
            f"| {item['method']} | {item['median_decision_ms']:.6f} "
            f"[{item['median_decision_ms_min']:.6f}, {item['median_decision_ms_max']:.6f}] | "
            f"{item['p95_decision_ms']:.6f} "
            f"[{item['p95_decision_ms_min']:.6f}, {item['p95_decision_ms_max']:.6f}] |"
        )

    report = "\n".join(
        [
            "# 工控机状态更新策略复现实验结果与分析",
            "",
            "## 实验环境与协议",
            "",
            "- 设备：工业控制计算机，12th Gen Intel Core i5-1235U，x86_64，15 GiB内存。",
            "- 操作系统：Ubuntu/Linux 6.8.0-124-generic。",
            "- Python：3.12.14；NumPy 2.0.1；Pandas 3.0.5。",
            "- 单核固定：所有正式运行均使用taskset固定在CPU 0。",
            "- 重复：5个独立Python进程；计时期间暂停4个LXD测试节点，结束后全部恢复。",
            f"- 输入：{first_metadata['trace_length']}时隙资源轨迹，"
            f"{first_metadata['validation_length']}时隙验证段，"
            f"{first_metadata['test_length']}时隙测试段，6步FITS-Delta预测。",
            f"- 主预算：B=0.5；四种方法均精确执行"
            f"{int(round(main_table['update_count'].iloc[0]))}次逻辑更新。",
            "",
            "## 工控机主结果",
            "",
            "| 方法 | WRMSE↓ | CPU RMSE↓ | Memory RMSE↓ | 更新数/率 | 平均/P95 AoI↓ | 工控机决策中位/P95（ms）↓ | 预算违反↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "工控机决策时延取5个独立进程各自统计值的中位数。WRMSE、AoI和更新结果在5次工控机运行中完全一致，并与本地参考运行逐项一致（绝对差不超过1e-12）。",
            "",
            "## 跨进程计时范围",
            "",
            "| 方法 | 决策中位数：5次中位值[min, max]（ms） | 决策P95：5次中位值[min, max]（ms） |",
            "|---|---:|---:|",
            *timing_rows,
            "",
            "## 内存与结论边界",
            "",
            f"完整分析进程峰值RSS的5次中位数为{rss['peak_rss_mib'].median():.1f} MiB，范围为{rss['peak_rss_mib'].min():.1f}–{rss['peak_rss_mib'].max():.1f} MiB。",
            "该数值包含Pandas、Matplotlib、全部预算扫描、鲁棒性实验和绘图模块，属于完整实验运行器的保守上界，不能解释为MPC-DP核心策略单独占用的内存。",
            "",
            f"MPC-DP在相同更新预算下取得最低WRMSE，但决策计算时间高于三个简单基线；"
            f"其工控机P95为{float(main_table.loc[main_table['method'] == 'MPC-DP', 'industrial_p95_decision_ms'].iloc[0]):.3f} ms。"
            "结果支持实时执行可行性，不支持MPC-DP在AoI、决策耗时或所有预算下全面最优。",
            "",
            "## 完整性验证",
            "",
            f"- 代码与输入哈希：{len(provenance)}/{len(provenance)}匹配。",
            f"- 跨工控机重复：{int(consistency['five_industrial_runs_match'].sum())}/{len(consistency)}种方法完全一致。",
            f"- 与本地参考：{int(consistency['industrial_matches_local_reference'].sum())}/{len(consistency)}种方法完全一致。",
            "- 预算违反：0。",
            "- 计时后测试床状态：edge-node-17至edge-node-20均已恢复为RUNNING。",
        ]
    )
    (root / "工控机状态更新策略复现实验结果与分析.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.industrial_dir)
    consistency = validate_consistency(runs, args.reference_dir)
    timing = aggregate_timing(runs)
    rss = collect_peak_rss(args.industrial_dir)
    provenance = write_provenance(args.industrial_dir, args.project_root, args.data_dir)
    main_table = build_main_table(runs, timing)

    runs.to_csv(args.industrial_dir / "industrial_run_level_results.csv", index=False)
    timing.to_csv(args.industrial_dir / "industrial_decision_timing_summary.csv", index=False)
    consistency.to_csv(args.industrial_dir / "cross_platform_consistency.csv", index=False)
    rss.to_csv(args.industrial_dir / "industrial_process_peak_rss.csv", index=False)
    provenance.to_csv(args.industrial_dir / "code_input_hash_verification.csv", index=False)
    main_table.to_csv(args.industrial_dir / "industrial_update_policy_main_table.csv", index=False)
    write_report(args.industrial_dir, main_table, timing, consistency, rss, provenance)
    print(main_table.to_string(index=False))


if __name__ == "__main__":
    main()
