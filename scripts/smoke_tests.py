"""Run input-free checks for this staging package.

These checks exercise imports, small protocol invariants, and CLI parsing.
They do not reproduce any manuscript result or launch a physical testbed.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DIRECT_TESTS = (
    "prediction_experiments/baseline_experiments/test_protocol.py",
    "prediction_experiments/baseline_experiments/test_all_models_unified.py",
    "raft/search_experiments/test_dynamic_search.py",
    "raft/testbed_deployment/test_node_agent.py",
    "raft/testbed_deployment/test_shared_update_policy_conformance.py",
)
ENTRY_POINTS = (
    "scripts/run_prediction_benchmark.py",
    "prediction_experiments/baseline_experiments/run_all_models_unified.py",
    "prediction_experiments/baseline_experiments/prepare_formal_prediction_bundle.py",
    "prediction_experiments/baseline_experiments/run_alibaba_disk_io_d3.py",
    "prediction_experiments/baseline_experiments/run_alibaba_disk_io_d3_five_models.py",
    "raft/update_experiments/run_update_experiments.py",
    "raft/update_experiments/run_disk_io_d3_industrial_latency.py",
    "raft/update_experiments/aggregate_disk_io_d3_industrial_latency.py",
    "raft/update_experiments/plot_disk_io_d3_compatibility.py",
    "raft/search_experiments/run_industrial_100node_discovery_suite.py",
    "raft/joint_experiments/run_deployment_equivalent_network_scale.py",
)


def run(label: str, args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    print(f"[RUN] {label}", flush=True)
    subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL if args[-1] == "--help" else None,
    )
    print(f"[PASS] {label}", flush=True)


def main() -> int:
    try:
        for relative in DIRECT_TESTS:
            run(Path(relative).name, [sys.executable, str(ROOT / relative)])

        update_test = ROOT / "raft/update_experiments/test_update_experiments.py"
        call_functions = (
            "import runpy, sys; "
            "scope = runpy.run_path(sys.argv[1]); "
            "checks = [value for name, value in scope.items() "
            "if name.startswith('test_') and callable(value)]; "
            "[check() for check in checks]; "
            "print(f'{len(checks)} update-policy checks passed')"
        )
        run("update-policy functions", [sys.executable, "-c", call_functions, str(update_test)])

        for relative in ENTRY_POINTS:
            run(f"{Path(relative).name} --help", [sys.executable, str(ROOT / relative), "--help"])
        run(
            "archived update import and CLI",
            [sys.executable, str(ROOT / "scripts/run_archived_update.py"), "--", "--help"],
        )
    except subprocess.CalledProcessError as exc:
        print(f"Smoke checks stopped: process exited with status {exc.returncode}.", file=sys.stderr)
        return 1
    print("Input-free smoke checks passed; manuscript results remain unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
