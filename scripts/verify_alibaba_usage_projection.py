"""Read-only row check for the local Alibaba two-column trace projection.

The archived prediction pipeline consumes ``usage_2018.csv`` with the column
labels ``2,3``. This tool checks whether those two columns correspond, in the
same order, to zero-based columns 2 and 3 of a local ``machine_usage.csv``.
It does not establish the upstream release identity or redistribution rights.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from itertools import zip_longest
from pathlib import Path


def numeric_value(value: str) -> Decimal | None:
    value = value.strip()
    if not value or value.lower() in {"nan", "na", "null"}:
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("non-numeric resource value") from exc


def verify(
    raw_path: Path,
    derived_path: Path,
    limit: int | None,
    allow_cpu_mean_fill: bool,
) -> int:
    with raw_path.open("r", encoding="utf-8", newline="") as raw_stream, derived_path.open(
        "r", encoding="utf-8", newline=""
    ) as derived_stream:
        raw_rows = csv.reader(raw_stream)
        derived_rows = csv.reader(derived_stream)
        if next(derived_rows, None) != ["2", "3"]:
            raise ValueError("derived header is not the expected 2,3")
        checked = 0
        missing_cpu = 0
        cpu_total = 0.0
        cpu_observed = 0
        cpu_fill: Decimal | None = None
        for row_number, (raw, derived) in enumerate(
            zip_longest(raw_rows, derived_rows), start=1
        ):
            if limit is not None and checked >= limit:
                break
            if raw is None or derived is None:
                raise ValueError(f"row count differs at row {row_number}")
            if len(raw) < 4 or len(derived) != 2:
                raise ValueError(f"unexpected CSV shape at row {row_number}")
            raw_cpu, raw_memory = numeric_value(raw[2]), numeric_value(raw[3])
            derived_cpu, derived_memory = numeric_value(derived[0]), numeric_value(derived[1])
            if raw_memory != derived_memory:
                raise ValueError(f"memory projection differs at row {row_number}")
            if raw_cpu is None and allow_cpu_mean_fill:
                if derived_cpu is None:
                    raise ValueError(f"CPU fill missing at row {row_number}")
                if cpu_fill is None:
                    cpu_fill = derived_cpu
                elif derived_cpu != cpu_fill:
                    raise ValueError(f"CPU fill differs at row {row_number}")
                missing_cpu += 1
            elif raw_cpu != derived_cpu:
                raise ValueError(f"CPU/memory projection differs at row {row_number}")
            if raw_cpu is not None:
                cpu_total += float(raw_cpu)
                cpu_observed += 1
            checked += 1
            if checked % 1_000_000 == 0:
                print(f"[PROGRESS] {checked:,} rows compared", flush=True)
    scope = "PREFIX ONLY" if limit is not None else "FULL FILE"
    print(f"[PASS] {scope}: {checked:,} aligned CPU/memory rows compared")
    if missing_cpu:
        observed_mean = cpu_total / cpu_observed
        print(
            f"[INFO] {missing_cpu} missing raw CPU values use a common derived fill "
            f"of {cpu_fill}; observed raw-CPU mean is {observed_mean:.12f}"
        )
        if limit is None and abs(float(cpu_fill) - observed_mean) > 1e-6:
            raise ValueError("common CPU fill does not match the full observed raw-CPU mean")
    print("This does not establish upstream trace identity or reuse permission.")
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_machine_usage", type=Path)
    parser.add_argument("derived_usage_2018", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="check a prefix only")
    parser.add_argument(
        "--allow-cpu-mean-fill",
        action="store_true",
        help="allow one shared fill for missing raw CPU values and verify its full-file mean",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        verify(
            args.raw_machine_usage,
            args.derived_usage_2018,
            args.limit,
            args.allow_cpu_mean_fill,
        )
    except (OSError, ValueError, csv.Error) as exc:
        print(f"[FAIL] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
