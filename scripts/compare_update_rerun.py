"""Compare archived and rerun state-update results, excluding CPU timing.

Decision-time measurements depend on the processor, OS and Python runtime;
this check tests algorithmic output equivalence, not latency equivalence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TABLES = {
    "raw_runs.csv": ("method", "budget", "seed"),
    "pareto_summary.csv": ("method", "budget"),
    "main_B0.5_summary.csv": ("method", "budget"),
    "selected_hyperparameters.csv": ("budget",),
    "calibration_details.csv": (
        "method", "budget", "seed", "calibration_type", "parameter", "horizon",
        "cpu_weight", "memory_weight",
    ),
    "eta_sensitivity.csv": ("method", "budget", "seed", "parameter"),
    "horizon_sensitivity.csv": ("method", "budget", "seed", "horizon"),
    "prediction_ablation_fixed_eta.csv": ("method", "budget", "seed", "variant"),
    "prediction_ablation.csv": ("method", "budget", "seed", "variant"),
    "update_loss_raw.csv": ("method", "budget", "seed", "loss_probability"),
    "update_loss_summary.csv": ("method", "loss_probability"),
    "weight_sensitivity.csv": (
        "method", "budget", "seed", "cpu_weight", "memory_weight",
    ),
}
TIMING_PREFIXES = ("decision_median_ms", "decision_p95_ms", "decision_mean_ms")


def compare(left_path: Path, right_path: Path, keys: tuple[str, ...]) -> bool:
    left = pd.read_csv(left_path).sort_values(list(keys)).reset_index(drop=True)
    right = pd.read_csv(right_path).sort_values(list(keys)).reset_index(drop=True)
    if left.shape != right.shape or set(left.columns) != set(right.columns):
        print(f"[FAIL] {left_path.name}: shape/column mismatch")
        return False
    if left.duplicated(list(keys)).any() or right.duplicated(list(keys)).any():
        print(f"[FAIL] {left_path.name}: non-unique comparison key")
        return False
    failed: list[str] = []
    checked = 0
    for column in left.columns:
        if column.startswith(TIMING_PREFIXES):
            continue
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            equal = np.isclose(
                a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                rtol=1e-10, atol=1e-10, equal_nan=True,
            ).all()
        else:
            equal = a.fillna("<NA>").astype(str).equals(b.fillna("<NA>").astype(str))
        if not equal:
            failed.append(column)
        checked += 1
    if failed:
        print(f"[FAIL] {left_path.name}: differing non-timing columns: {failed}")
        return False
    print(f"[PASS] {left_path.name}: {len(left)} keyed rows, {checked} non-timing columns")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    ok = True
    for name, keys in TABLES.items():
        left, right = args.archive / name, args.candidate / name
        if not left.is_file() or not right.is_file():
            print(f"[FAIL] missing {name}")
            ok = False
            continue
        ok = compare(left, right, keys) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
