"""Compare a new 100-node run with an archived run without altering either.

This is an output-equivalence check, not proof that unrecorded source files in
the archived run had the same bytes as the candidate reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TABLES = {
    "industrial_100node_discovery_raw.csv": (
        "experiment", "factor", "level", "malicious_rate", "repeat", "task", "pipeline"
    ),
    "industrial_100node_discovery_network.csv": (
        "experiment", "factor", "level", "malicious_rate", "pipeline", "repeat"
    ),
    "industrial_100node_discovery_summary.csv": (
        "experiment", "factor", "level", "malicious_rate", "pipeline"
    ),
}


def compare_table(archive: Path, candidate: Path, keys: tuple[str, ...]) -> bool:
    left = pd.read_csv(archive, low_memory=False)
    right = pd.read_csv(candidate, low_memory=False)
    if set(left.columns) != set(right.columns) or left.shape != right.shape:
        print(f"[FAIL] {archive.name}: columns or shape differ: {left.shape} vs {right.shape}")
        return False
    if left.duplicated(list(keys)).any() or right.duplicated(list(keys)).any():
        print(f"[FAIL] {archive.name}: comparison key is not unique")
        return False
    left = left.sort_values(list(keys)).reset_index(drop=True)
    right = right.sort_values(list(keys)).reset_index(drop=True)
    exact_columns: list[str] = []
    approximate_columns: list[tuple[str, int, float]] = []
    failed_columns: list[tuple[str, int, float | None]] = []
    for column in left.columns:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            close = np.isclose(x, y, rtol=1e-10, atol=1e-10, equal_nan=True)
            count = int((~close).sum())
            finite = np.isfinite(x) & np.isfinite(y)
            max_diff = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
            if count:
                failed_columns.append((column, count, max_diff))
            elif np.array_equal(x, y, equal_nan=True):
                exact_columns.append(column)
            else:
                approximate_columns.append((column, 0, max_diff))
        else:
            mismatch = int((a.fillna("<NA>").astype(str) != b.fillna("<NA>").astype(str)).sum())
            if mismatch:
                failed_columns.append((column, mismatch, None))
            else:
                exact_columns.append(column)
    if failed_columns:
        print(f"[FAIL] {archive.name}: {len(left)} rows; differing columns: {failed_columns}")
        return False
    print(
        f"[PASS] {archive.name}: {len(left)} keyed rows; "
        f"{len(exact_columns)} exact columns, {len(approximate_columns)} float-close columns"
    )
    if approximate_columns:
        print(f"       max absolute float differences: {approximate_columns}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    ok = True
    for name, keys in TABLES.items():
        if not (args.archive / name).is_file() or not (args.candidate / name).is_file():
            print(f"[FAIL] missing {name}")
            ok = False
            continue
        ok = compare_table(args.archive / name, args.candidate / name, keys) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
