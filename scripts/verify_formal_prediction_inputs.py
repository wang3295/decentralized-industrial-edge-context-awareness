"""Read-only check of the three frozen prediction inputs used downstream.

Usage: python scripts/verify_formal_prediction_inputs.py INPUT_DIRECTORY

This verifies bytes and CSV shape for the formal original-MPC-DP experiments.
It does not reproduce the FITS-Delta model or grant redistribution rights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import zip_longest
import math
from pathlib import Path
import sys


EXPECTED = {
    "value.csv": ("e2bf44333cd9c11cd15826f8a256c8928847e01ce72e82d900a54ddb281e845d", 14896, 2, False),
    "cpu_predictions.csv": ("64388fb14c0a53274d1ff31ae7444087ea137c42fa214dc09448beea3c5e06b9", 14896, 6, True),
    "memory_predictions.csv": ("920a65d6416b8fdf604d15ae1dfd147014566034560af9d6df5940d2aaa52a04", 14896, 6, True),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_csv(path: Path, rows_expected: int, columns_expected: int, has_header: bool) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if has_header:
            header = next(reader, [])
            expected_header = [f"step_{step}" for step in range(columns_expected)]
            if header != expected_header:
                errors.append("unexpected prediction header")
        for number, row in enumerate(reader, start=1):
            count += 1
            if len(row) != columns_expected:
                errors.append(f"row {number} has {len(row)} instead of {columns_expected} columns")
                continue
            try:
                if not all(math.isfinite(float(value)) for value in row):
                    errors.append(f"row {number} has non-finite values")
            except ValueError:
                errors.append(f"row {number} has a nonnumeric value")
            if len(errors) >= 5:
                break
    if count != rows_expected and len(errors) < 5:
        errors.append(f"found {count} data rows instead of {rows_expected}")
    return count, errors


def check_step_zero_alignment(directory: Path) -> list[str]:
    """The first prediction column must reproduce the corresponding observed value."""
    errors: list[str] = []
    with (directory / "value.csv").open(encoding="utf-8-sig", newline="") as values_file, \
        (directory / "cpu_predictions.csv").open(encoding="utf-8-sig", newline="") as cpu_file, \
        (directory / "memory_predictions.csv").open(encoding="utf-8-sig", newline="") as memory_file:
        values = csv.reader(values_file)
        cpu = csv.reader(cpu_file)
        memory = csv.reader(memory_file)
        next(cpu)
        next(memory)
        for number, rows in enumerate(zip_longest(values, cpu, memory), start=1):
            observed, cpu_prediction, memory_prediction = rows
            if observed is None or cpu_prediction is None or memory_prediction is None:
                errors.append(f"row {number}: files have different row counts")
                break
            for channel, actual, prediction in (
                ("CPU", observed[0], cpu_prediction[0]),
                ("memory", observed[1], memory_prediction[0]),
            ):
                if not math.isclose(float(actual), float(prediction), rel_tol=1e-7, abs_tol=1e-7):
                    errors.append(f"row {number}: {channel} step_0 differs from value.csv")
                    if len(errors) >= 5:
                        return errors
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()
    failures: list[str] = []
    for name, (expected_hash, rows_expected, columns_expected, has_header) in EXPECTED.items():
        path = args.input_dir / name
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        hash_ok = file_hash(path) == expected_hash
        if not hash_ok:
            failures.append(f"{name}: SHA-256 differs from the formal run")
        try:
            count, errors = check_csv(path, rows_expected, columns_expected, has_header)
        except (OSError, UnicodeError, csv.Error) as exc:
            failures.append(f"{name}: could not parse CSV ({type(exc).__name__})")
            continue
        failures.extend(f"{name}: {error}" for error in errors)
        if hash_ok and not errors:
            print(f"[PASS] {name}: SHA-256 and {count} x {columns_expected} CSV shape match")
    if not failures:
        try:
            failures.extend(check_step_zero_alignment(args.input_dir))
        except (OSError, UnicodeError, csv.Error, ValueError, IndexError) as exc:
            failures.append(f"cross-file step_0 check could not complete ({type(exc).__name__})")
        if not failures:
            print("[PASS] CPU/memory step_0 values match value.csv row by row")
    for failure in failures:
        print(f"[FAIL] {failure}")
    if failures:
        print("FORMAL INPUT VERIFICATION FAILED; no files were changed.")
        return 1
    print("FORMAL INPUT BYTES, SHAPES AND ALIGNMENT VERIFIED; upstream provenance and sharing rights remain separate checks.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OSError as exc:
        print(f"Verification could not complete: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2)
