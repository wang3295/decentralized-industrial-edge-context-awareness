"""Read-only safety check for this repository tree.

This is not a scientific reproduction test or a substitute for human review.
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "CITATION.cff",
    "data/README.md",
    "data/formal_inputs/README.md",
    "data/formal_inputs/manifest.json",
    "scripts/verify_formal_prediction_inputs.py",
    "scripts/smoke_tests.py",
    "scripts/synthetic_integration_check.py",
    "scripts/compare_scale_rerun.py",
    "scripts/run_archived_update.py",
    "scripts/compare_update_rerun.py",
    "REPRODUCE.md",
    "formal_runs/100node_validated_reconstruction/README.md",
    "formal_runs/update_main_20260914/source_snapshot/update_policies.py",
    "prediction_experiments/baseline_experiments/run_all_models_unified.py",
    "raft/search_experiments/run_industrial_100node_discovery_suite.py",
)
FORBIDDEN_PARTS = {"internal_review", ".git", "__pycache__", ".pytest_cache"}
FORBIDDEN_SUFFIXES = {".csv", ".pt", ".pth", ".ckpt", ".pem", ".key", ".pfx", ".pyc"}
SCANNED_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".ps1", ".cff"}
PRIVATE_MATERIAL = re.compile(
    r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"
    r"|github_pat_[A-Za-z0-9_]+"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|\b(?:192\.168\.|10\.\d+\.|172\.(?:1[6-9]|2\d|3[01])\.)\d+\.\d+\b"
    r"|(?<![A-Za-z])[A-Za-z]:[\\/]"
    r"|/home/td/",
    re.IGNORECASE,
)


def main() -> int:
    problems: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            problems.append(f"required file missing: {relative}")
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts:
            continue
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            problems.append(f"private or transient directory: {relative.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name == ".env":
            problems.append(f"data, weight, key, or transient file: {relative.as_posix()}")
        if path.suffix.lower() in SCANNED_SUFFIXES and path != Path(__file__).resolve():
            content = path.read_text(encoding="utf-8", errors="replace")
            if PRIVATE_MATERIAL.search(content):
                problems.append(f"manual review needed for path/address/key pattern: {relative.as_posix()}")
    if (ROOT / "LICENSE").exists():
        problems.append("LICENSE is present although the authors selected no software licence")
    if problems:
        for problem in problems:
            print(f"[CHECK] {problem}")
        print(f"Publication safety audit needs review: {len(problems)} finding(s).")
        return 1
    print("Publication safety audit passed: no listed private artifacts or patterns found.")
    print("No software licence is attached by author choice; if published, this is visible source, not open source.")
    print("This check does not establish independent reproduction of manuscript results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
