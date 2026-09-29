from __future__ import annotations

import argparse
import multiprocessing as mp
import time


def cpu_worker(deadline: float) -> None:
    value = 1.000001
    while time.monotonic() < deadline:
        value = (value * 1.0000001 + 0.0000003) % 97.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded CPU/memory load for the edge testbed")
    parser.add_argument("--cpu-workers", type=int, default=0, choices=range(0, 3))
    parser.add_argument("--memory-mb", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()
    if not 0 <= args.memory_mb <= 512:
        parser.error("--memory-mb must be between 0 and 512")
    if not 0.1 <= args.duration <= 300.0:
        parser.error("--duration must be between 0.1 and 300 seconds")
    return args


def main() -> None:
    args = parse_args()
    memory = bytearray(args.memory_mb * 1024 * 1024)
    if memory:
        memory[::4096] = b"x" * len(memory[::4096])
    deadline = time.monotonic() + args.duration
    workers = [mp.Process(target=cpu_worker, args=(deadline,)) for _ in range(args.cpu_workers)]
    for worker in workers:
        worker.start()
    while time.monotonic() < deadline:
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    for worker in workers:
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
