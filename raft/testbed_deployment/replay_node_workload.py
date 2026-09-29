from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time


def total_memory_mb() -> float:
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return page_size * pages / (1024 * 1024)


def load_schedule(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                duration = float(row["duration_s"])
                cpu_percent = float(row["cpu_percent"])
                memory_mb = int(row["memory_mb"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid schedule row {line_number}: {row}") from exc
            if not 0.1 <= duration <= 3600:
                raise ValueError(f"duration_s out of range at row {line_number}")
            if not 0 <= cpu_percent <= 100:
                raise ValueError(f"cpu_percent out of range at row {line_number}")
            if memory_mb < 0:
                raise ValueError(f"memory_mb must be non-negative at row {line_number}")
            rows.append(
                {
                    "step": int(row.get("step", len(rows))),
                    "label": row.get("label", "unspecified"),
                    "duration_s": duration,
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                }
            )
    if not rows:
        raise ValueError("schedule is empty")
    return rows


def cpu_worker(target_percent: mp.Value, stopped: mp.Event, quantum_s: float) -> None:
    value = 1.000001
    while not stopped.is_set():
        cycle_start = time.perf_counter()
        with target_percent.get_lock():
            busy_s = quantum_s * target_percent.value / 100.0
        busy_until = cycle_start + busy_s
        while time.perf_counter() < busy_until and not stopped.is_set():
            value = (value * 1.0000001 + 0.0000003) % 97.0
        remaining = quantum_s - (time.perf_counter() - cycle_start)
        if remaining > 0:
            stopped.wait(remaining)


def write_log(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a deterministic CPU/memory workload inside one edge VM"
    )
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--start-at-epoch", type=float)
    parser.add_argument("--max-memory-fraction", type=float, default=0.35)
    parser.add_argument("--cpu-quantum-ms", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not 0.05 <= args.max_memory_fraction <= 0.8:
        parser.error("--max-memory-fraction must be between 0.05 and 0.8")
    if not 10 <= args.cpu_quantum_ms <= 1000:
        parser.error("--cpu-quantum-ms must be between 10 and 1000")
    return args


def main() -> None:
    args = parse_args()
    schedule = load_schedule(args.schedule)
    memory_limit_mb = total_memory_mb() * args.max_memory_fraction
    largest_request = max(int(row["memory_mb"]) for row in schedule)
    if largest_request > memory_limit_mb:
        raise ValueError(
            f"schedule requests {largest_request} MB, exceeding the safety limit "
            f"of {memory_limit_mb:.1f} MB"
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "steps": len(schedule),
                    "duration_s": sum(float(row["duration_s"]) for row in schedule)
                    * args.repeat,
                    "largest_memory_mb": largest_request,
                    "memory_limit_mb": round(memory_limit_mb, 2),
                }
            )
        )
        return

    if args.start_at_epoch is not None:
        delay = args.start_at_epoch - time.time()
        if delay > 0:
            time.sleep(delay)

    stopped = mp.Event()
    target_percent = mp.Value("d", 0.0)
    worker = mp.Process(
        target=cpu_worker,
        args=(target_percent, stopped, args.cpu_quantum_ms / 1000.0),
        daemon=True,
    )

    def stop_handler(_signum, _frame) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    memory_block = bytearray()
    worker.start()
    try:
        with args.log.open("a", encoding="utf-8") as handle:
            write_log(
                handle,
                {
                    "event": "start",
                    "timestamp": time.time(),
                    "schedule": str(args.schedule),
                    "total_memory_mb": round(total_memory_mb(), 2),
                },
            )
            for repetition in range(args.repeat):
                for row in schedule:
                    if stopped.is_set():
                        break
                    memory_block = bytearray()
                    gc.collect()
                    memory_mb = int(row["memory_mb"])
                    memory_block = bytearray(memory_mb * 1024 * 1024)
                    if memory_block:
                        memory_block[::4096] = b"x" * len(memory_block[::4096])
                    with target_percent.get_lock():
                        target_percent.value = float(row["cpu_percent"])
                    write_log(
                        handle,
                        {
                            "event": "step",
                            "timestamp": time.time(),
                            "repetition": repetition,
                            **row,
                        },
                    )
                    stopped.wait(float(row["duration_s"]))
                if stopped.is_set():
                    break
            write_log(handle, {"event": "stop", "timestamp": time.time()})
    finally:
        with target_percent.get_lock():
            target_percent.value = 0.0
        stopped.set()
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2.0)
        del memory_block


if __name__ == "__main__":
    main()
