"""Read-only reconciliation of the final 20-node, five-block result files.

Usage: python scripts/verify_physical20_summary.py PATH_TO_FINAL_RUN

This checks archived CSV/JSON consistency. It does not rerun the deployment or
establish that a sanitized public runner reproduces the measured latencies.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one observation")
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def close(actual: float, expected: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)


def verify(root: Path) -> tuple[list[str], list[str], list[tuple[str, int, float, float, float]]]:
    formal = root / "formal"
    errors: list[str] = []
    warnings: list[str] = []
    table: list[tuple[str, int, float, float, float]] = []

    completion = read_json(root / "completion.json")
    manifest = read_json(formal / "protocol_manifest.json")
    validation = read_json(formal / "validation_and_descriptive_results.json")
    audit = read_json(formal / "evidence_audit.json")
    clock = read_json(root / "clock_formal.json")
    requests = read_csv(formal / "physical20_requests.csv")
    blocks = read_csv(formal / "physical20_blocks.csv")
    summaries = read_csv(formal / "physical20_summary.csv")
    instrument_requests = read_csv(root / "instrumentation_check" / "physical20_requests.csv")

    settings = manifest["settings"]
    expected_blocks = int(settings["blocks"])
    expected_trials = int(settings["trials_per_block"])
    methods = list(settings["methods"])
    expected_requests = expected_blocks * len(methods) * expected_trials
    if (expected_blocks, len(methods), expected_trials) != (5, 5, 100):
        errors.append("protocol design differs from the final 5 x 5 x 100 manuscript run")
    if len(instrument_requests) != 100:
        errors.append("instrumentation check does not contain exactly 100 separate requests")
    if len(requests) != expected_requests or len(requests) != 2500:
        errors.append("formal request count is not 2,500")
    if not completion.get("accepted") or completion.get("formal_requests") != 2500:
        errors.append("completion record is not an accepted 2,500-request run")
    if completion.get("formal_blocks") != 5:
        errors.append("completion record does not report five formal blocks")
    if any(code != 0 for code in completion.get("steps", {}).values()):
        errors.append("a supervisor step has a nonzero exit code")
    if clock.get("status") != "restored" or clock.get("restored_epoch") is None:
        errors.append("formal clock restoration is not recorded")

    checks = validation.get("checks", {})
    if not validation.get("passed") or not checks or not all(checks.values()):
        errors.append("formal validation checks did not all pass")
    if not audit.get("passed") or audit.get("nodes") != 20 or audit.get("raw_responses") != 2500:
        errors.append("formal evidence audit is not complete")
    for name in (
        "unmatched_event_streams",
        "unmatched_snapshots",
        "unmatched_responses",
        "missing_workload_runs",
    ):
        if audit.get(name):
            errors.append(f"evidence audit contains {name}")
    preserved = audit.get("preserved_parse_errors", {})
    parse_error_count = sum(preserved.values()) if isinstance(preserved, dict) else 0
    if parse_error_count:
        warnings.append(f"{parse_error_count} preserved, unparsed node-log lines require disclosure review")

    group_sizes = Counter((row["block"], row["method"]) for row in requests)
    expected_groups = {(str(block), method) for block in range(1, 6) for method in methods}
    if set(group_sizes) != expected_groups or set(group_sizes.values()) != {100}:
        errors.append("formal requests are not exactly 25 groups of 100")
    block_groups = Counter((row["block"], row["method"]) for row in blocks)
    if set(block_groups) != expected_groups or set(block_groups.values()) != {1}:
        errors.append("block-metrics file does not have exactly one row per formal group")
    summary_by_method = {row["method"]: row for row in summaries}
    if len(summaries) != 5 or set(summary_by_method) != set(methods):
        errors.append("summary file does not have one row for each of the five methods")

    requests_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    blocks_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in requests:
        requests_by_method[row["method"]].append(row)
    for row in blocks:
        blocks_by_method[row["method"]].append(row)
    for method in methods:
        group = requests_by_method[method]
        block_rows = blocks_by_method[method]
        summary = summary_by_method.get(method)
        if len(group) != 500 or len(block_rows) != 5 or summary is None:
            errors.append(f"incomplete records for {method}")
            continue
        successes = [row for row in group if row["success"].lower() in {"1", "true"}]
        rate = len(successes) / 5  # percent of 500 requests
        p95 = percentile([float(row["decision_latency_ms"]) for row in successes], 0.95)
        messages = sum(
            int(row["update_messages_attempted"])
            + int(row["search_forwards_sent"])
            + int(row["confirmations_attempted"])
            for row in block_rows
        ) / 500
        if int(summary["requests"]) != 500:
            errors.append(f"summary request count differs for {method}")
        if not close(rate, float(summary["success_rate_percent"])):
            errors.append(f"summary discovery rate differs for {method}")
        if not close(p95, float(summary["p95_success_latency_ms"])):
            errors.append(f"summary P95 decision latency differs for {method}")
        if not close(messages, float(summary["total_messages_per_request"])):
            errors.append(f"summary application-message count differs for {method}")
        if sum(int(row["successes"]) for row in block_rows) != len(successes):
            errors.append(f"block-level success total differs for {method}")
        table.append((method, len(successes), rate, p95, messages))

    source = formal / "source"
    if (source / "node_agent.py").is_file():
        if sha256(source / "node_agent.py") != manifest.get("agent_sha256"):
            errors.append("archived node-agent hash differs from protocol manifest")
    else:
        warnings.append("node-agent source snapshot is absent")
    if (source / "update_policies.py").is_file():
        if sha256(source / "update_policies.py") != manifest.get("core_sha256"):
            errors.append("archived update-policy hash differs from protocol manifest")
    else:
        warnings.append("update-policy source snapshot is absent")
    missing_sources = [name for name in manifest.get("sources", {}) if not (source / name).is_file()]
    if missing_sources:
        warnings.append(f"{len(missing_sources)} manifest-listed source files are absent from the archived snapshot")
    warnings.append("an indirectly imported deployment helper lacks an archived run-time hash")
    return errors, warnings, table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="root containing completion.json and formal/")
    args = parser.parse_args()
    try:
        errors, warnings, table = verify(args.result_dir)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(f"Verification could not complete: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for method, successes, rate, p95, messages in table:
        print(f"{method}: {successes}/500; {rate:.2f}%; P95 {p95:.2f} ms; {messages:.2f} messages/request")
    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[FAIL] {error}")
    if errors:
        print(f"ARCHIVED RESULTS NOT VERIFIED: {len(errors)} discrepancy/discrepancies.")
        return 1
    print("ARCHIVED CSV/JSON CONSISTENCY VERIFIED; deployment reproduction not established.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
