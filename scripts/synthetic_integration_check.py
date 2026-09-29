"""Run a tiny, deterministic 100-node integration check with synthetic inputs.

This checks the input schema, MPC-DP replay, search runner, and output audit.
It is deliberately NOT a reproduction of any manuscript result. Temporary
synthetic CSVs and outputs are removed when the check exits.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "formal_runs/100node_validated_reconstruction/search_experiments"
    / "run_industrial_100node_discovery_suite.py"
)


def write_inputs(directory: Path) -> None:
    actual: list[tuple[float, float]] = []
    for slot in range(500):
        cpu = 30.0 + (slot % 29) * 0.7
        memory = 35.0 + (slot % 23) * 0.5
        actual.append((cpu, memory))

    with (directory / "value.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(actual)
    for dimension, filename in enumerate(("cpu_predictions.csv", "memory_predictions.csv")):
        with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"step_{step}" for step in range(6)])
            for cpu, memory in actual:
                current = (cpu, memory)[dimension]
                writer.writerow([current] * 6)


def main() -> int:
    with TemporaryDirectory(prefix="edge_artifact_synthetic_") as temporary:
        workspace = Path(temporary)
        inputs = workspace / "inputs"
        output = workspace / "output"
        inputs.mkdir()
        write_inputs(inputs)
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [
            sys.executable, "-B", str(RUNNER),
            "--data-dir", str(inputs),
            "--output-dir", str(output),
            "--repeats", "1",
            "--tasks-per-network", "2",
            "--trace-length", "282",
            "--workers", "1",
            "--sections", "protocol,hyperparameter",
        ]
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
        if validation.get("status") != "PASS":
            raise AssertionError(f"Synthetic integration validation failed: {validation}")
        if validation.get("actual_raw_rows") != validation.get("expected_raw_rows"):
            raise AssertionError("Synthetic integration raw row count differs from expected")
        if validation.get("independent_networks_per_condition") != 1:
            raise AssertionError("Synthetic integration network count is wrong")
    print("Synthetic integration check passed; no manuscript result was reproduced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
