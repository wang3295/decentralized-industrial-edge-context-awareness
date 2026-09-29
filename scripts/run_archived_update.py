"""Run the archived update source with its archived policy-core bytes.

The historical source snapshot retained two Python files but not their
original package directory layout. This wrapper binds the archived
`update_policies.py` to the import name expected by the unchanged archived
`run_update_experiments.py`. It does not modify either archived file.
Pass normal runner arguments after a standalone `--`.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import runpy
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "formal_runs/update_main_20260914/source_snapshot"
CORE = SNAPSHOT / "update_policies.py"
RUNNER = SNAPSHOT / "run_update_experiments.py"


def bind_archived_core() -> None:
    package = types.ModuleType("edgebench_core")
    package.__path__ = [str(SNAPSHOT)]
    sys.modules["edgebench_core"] = package
    spec = importlib.util.spec_from_file_location("edgebench_core.update_policies", CORE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load archived update core: {CORE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    package.update_policies = module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.runner_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    bind_archived_core()
    sys.argv = [str(RUNNER), *forwarded]
    runpy.run_path(str(RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
