"""Read-only audit of the archived 100-node nominal discovery comparison.

Usage: python scripts/verify_100node_nominal.py PATH_TO_FORMAL_RESULT_DIR

This reaggregates nominal raw requests into 25 independent network rows and
compares them with the archived summary. It does not rerun graph simulations,
reconstruct unarchived indirect dependency versions or verify every robustness
and sensitivity condition.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


RESULT_SHA256 = {
    "industrial_100node_discovery_raw.csv": "8d923d320854dc5b9373167b5c70282b822c185107e533949f2a9452d877778f",
    "industrial_100node_discovery_network.csv": "356b36ef60c62d1aabd5b6309d76ef7272c50334993c8ab9346e95ae798e3822",
    "industrial_100node_discovery_summary.csv": "2eb5b0e3e5e732b9260f4b83fbb36e363be41166f4b3230c2e4abb414eb7a6e5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nominal(row: dict[str, str]) -> bool:
    return (
        row["experiment"] == "protocol_main"
        and row["factor"] == "malicious_rate"
        and math.isclose(float(row["malicious_rate"]), 0.0, abs_tol=1e-12)
        and math.isclose(float(row["level"]), 0.0, abs_tol=1e-12)
    )


def read_nominal(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if nominal(row)]


def close(a: float, b: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=tolerance)


def verify(root: Path) -> tuple[list[str], list[str], list[tuple[str, float, float]]]:
    errors: list[str] = []
    warnings: list[str] = []
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    if (
        config.get("num_nodes") != 100
        or config.get("network_repeats") != 25
        or config.get("tasks_per_network") != 120
        or config.get("threshold_lambda") != 0.0
    ):
        errors.append("run metadata is not the formal 100-node original-MPC-DP design")
    if validation.get("status") != "PASS":
        errors.append("formal validation status is not PASS")
    if validation.get("expected_raw_rows") != validation.get("actual_raw_rows"):
        errors.append("raw-row count differs from formal validation expectation")
    if validation.get("expected_network_rows") != validation.get("actual_network_rows"):
        errors.append("network-row count differs from formal validation expectation")
    for name, expected in RESULT_SHA256.items():
        if sha256(root / name) != expected:
            errors.append(f"archived result SHA-256 differs: {name}")
    for relative, expected in metadata.get("source_sha256", {}).items():
        path = root / "source_snapshot" / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"formal source snapshot missing or changed: {relative}")
    warnings.append("two indirectly imported helper versions lack formal-run hashes")

    raw = read_nominal(root / "industrial_100node_discovery_raw.csv")
    network = read_nominal(root / "industrial_100node_discovery_network.csv")
    summary = read_nominal(root / "industrial_100node_discovery_summary.csv")
    if len(raw) != 5 * 25 * 120 or len(network) != 5 * 25 or len(summary) != 5:
        errors.append("nominal records are not five methods x 25 networks x 120 requests")

    raw_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    network_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    summary_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        raw_groups[(row["pipeline"], int(row["repeat"]))].append(row)
    for row in network:
        network_groups[(row["pipeline"], int(row["repeat"]))].append(row)
    for row in summary:
        summary_groups[row["pipeline"]].append(row)
    methods = {"centralized_oracle", "expanding_ring", "gossip", "proposed", "random_walk"}
    expected_groups = {(method, repeat) for method in methods for repeat in range(25)}
    if set(raw_groups) != expected_groups or set(network_groups) != expected_groups:
        errors.append("nominal per-network grouping differs from the expected design")
    if set(summary_groups) != methods or any(len(rows) != 1 for rows in summary_groups.values()):
        errors.append("nominal summary methods are incomplete or duplicated")

    output: list[tuple[str, float, float]] = []
    for method in sorted(methods):
        rates: list[float] = []
        messages: list[float] = []
        for repeat in range(25):
            tasks = raw_groups.get((method, repeat), [])
            rows = network_groups.get((method, repeat), [])
            if len(tasks) != 120 or len(rows) != 1:
                errors.append(f"incomplete nominal network records for {method}")
                continue
            rate = sum(int(row["success"]) for row in tasks) / 120
            message_mean = sum(float(row["search_messages"]) for row in tasks) / 120
            if not close(rate, float(rows[0]["success_rate"])):
                errors.append(f"network success rate differs for {method}, repeat {repeat}")
            if not close(message_mean, float(rows[0]["mean_discovery_messages"])):
                errors.append(f"network message mean differs for {method}, repeat {repeat}")
            rates.append(rate)
            messages.append(message_mean)
        if len(rates) != 25 or method not in summary_groups:
            continue
        observed_rate = sum(rates) / 25 * 100
        observed_messages = sum(messages) / 25
        reported = summary_groups[method][0]
        if int(reported["networks"]) != 25:
            errors.append(f"summary does not use 25 networks for {method}")
        if not close(observed_rate, float(reported["success_percent"])):
            errors.append(f"summary success rate differs for {method}")
        if not close(observed_messages, float(reported["mean_discovery_messages"])):
            errors.append(f"summary message mean differs for {method}")
        output.append((method, observed_rate, observed_messages))
    return errors, warnings, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    try:
        errors, warnings, table = verify(args.result_dir)
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        print(f"Verification could not complete: {type(exc).__name__}", file=sys.stderr)
        return 2
    for method, rate, messages in table:
        print(f"{method}: {rate:.2f}% discovery; {messages:.2f} discovery messages/request")
    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[FAIL] {error}")
    if errors:
        print(f"ARCHIVED NOMINAL COMPARISON NOT VERIFIED: {len(errors)} discrepancy/discrepancies.")
        return 1
    print("ARCHIVED NOMINAL RAW/NETWORK/SUMMARY CONSISTENCY VERIFIED; simulation rerun not established.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
