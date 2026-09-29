"""Run the archived nine-model benchmark with explicit local trace paths.

This wrapper changes only dataset locations. It imports the unmodified
`run_all_models_unified.py` implementation and passes all other CLI options
through to that program. It does not download or redistribute third-party
traces and does not guarantee identical hardware timing or model checkpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "prediction_experiments/baseline_experiments/run_all_models_unified.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alibaba-csv", type=Path, required=True)
    parser.add_argument("--gwa-csv", type=Path, required=True)
    parser.add_argument("--google-csv", type=Path, required=True)
    parser.add_argument(
        "runner_args", nargs=argparse.REMAINDER,
        help="Arguments for run_all_models_unified.py after a standalone '--'",
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("public_unified_benchmark", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import benchmark runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    replacements = {
        "Alibaba": args.alibaba_csv,
        "GWA-T-12": args.gwa_csv,
        "Google": args.google_csv,
    }
    for name, provided in replacements.items():
        path = provided.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        original = module.BASE.DATASETS[name]
        module.BASE.DATASETS[name] = module.BASE.DatasetSpec(
            name=original.name, path=path, header=original.header,
            columns=original.columns,
        )
    forwarded = args.runner_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    sys.argv = [str(RUNNER), *forwarded]
    module.main()


if __name__ == "__main__":
    main()
