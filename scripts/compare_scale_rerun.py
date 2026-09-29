"""Compare 50/100/150-node rerun tables against an accepted archive.

This compares parsed, keyed values. CSV byte hashes can differ when pandas
or the operating system serializes floating-point values differently.
It does not verify that the historical run loaded identical source bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_100node_rerun import compare_table


TABLES = {
    "deployment_equivalent_raw.csv": (
        "num_nodes", "malicious_rate", "repeat", "task", "pipeline",
    ),
    "deployment_equivalent_network.csv": (
        "num_nodes", "malicious_rate", "repeat", "pipeline",
    ),
    "deployment_equivalent_summary.csv": ("num_nodes", "pipeline"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    ok = True
    for root in (args.archive, args.candidate):
        validation_file = root / "validation.json"
        if not validation_file.is_file():
            print(f"[FAIL] missing {validation_file}")
            ok = False
            continue
        validation = json.loads(validation_file.read_text(encoding="utf-8"))
        if validation.get("status") != "PASS":
            print(f"[FAIL] validation did not pass: {validation_file}")
            ok = False
    for name, keys in TABLES.items():
        left, right = args.archive / name, args.candidate / name
        if not left.is_file() or not right.is_file():
            print(f"[FAIL] missing comparison table {name}")
            ok = False
            continue
        ok = compare_table(left, right, keys) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
