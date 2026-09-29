from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web
import numpy as np
import psutil
import torch
import torch.nn as nn
import yaml


RAFT_ROOT = Path(__file__).resolve().parents[1]
if str(RAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(RAFT_ROOT))

from edgebench_core.update_policies import (  # noqa: E402
    RepeatedEpochBudgetGuard as TokenBudget,
    mpc_dp_first_action,
    runtime_update_proposal,
    weighted_squared_error,
)


RESOURCE_KEYS = ("cpu", "memory")


def now_ms() -> float:
    return time.time() * 1000.0


def clamp_resource(values: np.ndarray | list[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 100.0)


def qualifies(available: np.ndarray, demand: np.ndarray) -> bool:
    return bool(np.all(np.asarray(available) >= np.asarray(demand)))


def direct_trust(alpha: float, beta: float) -> float:
    return float(alpha / (alpha + beta))


def extend_path_trust(previous: float | None, edge_trust: float, gamma: float) -> float:
    if previous is None:
        return float(edge_trust)
    return float(gamma * math.sqrt(max(previous, 0.0) * max(edge_trust, 0.0)))


def can_reach_one_more_hop(path: list[str], max_hops: int) -> bool:
    """Whether a candidate one edge beyond ``path`` remains within D_max."""
    return len(path) - 1 < max_hops


class FITSModel(nn.Module):
    """Compact copy of the manuscript FITS architecture."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, cut_freq: int = 24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.dominance_freq = cut_freq
        self.length_ratio = (seq_len + pred_len) / seq_len
        output_freq = int(cut_freq * self.length_ratio)
        self.freq_upsampler = nn.Linear(cut_freq, output_freq, dtype=torch.cfloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x, dim=1, keepdim=True)
        centered = x - x_mean
        x_var = torch.var(centered, dim=1, keepdim=True) + 1e-5
        normalized = centered / torch.sqrt(x_var)
        spectrum = torch.fft.rfft(normalized, dim=1)
        spectrum[:, self.dominance_freq :] = 0
        spectrum = spectrum[:, : self.dominance_freq, :]
        upsampled = self.freq_upsampler(spectrum.permute(0, 2, 1)).permute(0, 2, 1)
        target_length = self.seq_len + self.pred_len
        full_spectrum = torch.zeros(
            [upsampled.size(0), target_length // 2 + 1, self.channels],
            dtype=torch.cfloat,
            device=x.device,
        )
        full_spectrum[:, : upsampled.size(1), :] = upsampled
        signal = torch.fft.irfft(full_spectrum, n=target_length, dim=1) * self.length_ratio
        output = signal * torch.sqrt(x_var) + x_mean
        return output[:, -self.pred_len :, :]


class FITSDeltaModel(nn.Module):
    """FITS with first-difference modeling and cumulative reconstruction."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, cut_freq: int = 24):
        super().__init__()
        self.base = FITSModel(seq_len, pred_len, channels, cut_freq=cut_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        differences = torch.zeros_like(x)
        differences[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        predicted_differences = self.base(differences)
        return x[:, -1:, :] + torch.cumsum(predicted_differences, dim=1)


class FITSForecaster:
    def __init__(self, checkpoint: Path):
        checkpoint = Path(checkpoint)
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = checkpoint_data["config"]
        self.seq_len = int(cfg["seq_len"])
        self.pred_len = int(cfg["pred_len"])
        self.mean = np.asarray(checkpoint_data["mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint_data["std"], dtype=np.float32)
        self.model_key = str(checkpoint_data.get("model_key", "fits")).lower()
        self.model_label = str(checkpoint_data.get("model_label", self.model_key))
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        state = checkpoint_data["model_state"]
        is_delta = self.model_key == "fits_delta" or any(
            str(key).startswith("base.") for key in state
        )
        model_type = FITSDeltaModel if is_delta else FITSModel
        self.model = model_type(
            self.seq_len,
            self.pred_len,
            channels=2,
            cut_freq=int(cfg["fits_cut_freq"]),
        )
        self.model.load_state_dict(state)
        self.model.eval()
        torch.set_num_threads(min(2, max(1, psutil.cpu_count(logical=True) or 1)))

    def predict(self, history: list[np.ndarray]) -> tuple[np.ndarray, float]:
        if not history:
            raise ValueError("history must not be empty")
        values = np.asarray(history[-self.seq_len :], dtype=np.float32)
        if len(values) < self.seq_len:
            pad = np.repeat(values[:1], self.seq_len - len(values), axis=0)
            values = np.concatenate([pad, values], axis=0)
        scaled = (values - self.mean) / self.std
        sample = torch.from_numpy(scaled).unsqueeze(0)
        started = time.perf_counter_ns()
        with torch.inference_mode():
            predicted_scaled = self.model(sample).squeeze(0).numpy()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        predicted = predicted_scaled * self.std + self.mean
        return clamp_resource(predicted), float(elapsed_ms)


@dataclass
class CachedState:
    node_id: str
    base_url: str
    timestamp_ms: float
    received_ms: float
    sequence: int
    utilization: np.ndarray
    availability: np.ndarray
    forecast_utilization: np.ndarray

    def public(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "base_url": self.base_url,
            "timestamp_ms": self.timestamp_ms,
            "received_ms": self.received_ms,
            "age_ms": max(0.0, now_ms() - self.received_ms),
            "sequence": self.sequence,
            "utilization": dict(zip(RESOURCE_KEYS, self.utilization.tolist())),
            "availability": dict(zip(RESOURCE_KEYS, self.availability.tolist())),
            "forecast_utilization": self.forecast_utilization.tolist(),
        }


class NodeAgent:
    def __init__(self, config_path: Path, model_path: Path):
        self.config_path = config_path
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.node_id = str(self.config["node_id"])
        self.host = str(self.config.get("listen_host", "0.0.0.0"))
        self.port = int(self.config.get("listen_port", 8801))
        self.advertise_url = str(self.config.get("advertise_url", "")).rstrip("/")
        if not self.advertise_url:
            raise ValueError("config must define advertise_url for distributed confirmation")

        self.neighbors = {
            str(item["node_id"]): str(item["base_url"]).rstrip("/")
            for item in self.config.get("business_neighbors", [])
        }
        if not self.neighbors:
            raise ValueError("at least one business neighbor is required")

        monitor_cfg = self.config.get("monitor", {})
        update_cfg = self.config.get("update", {})
        search_cfg = self.config.get("search", {})
        trust_cfg = self.config.get("trust", {})
        self.interval = float(monitor_cfg.get("interval_seconds", 1.0))
        self.budget = float(update_cfg.get("budget", 0.5))
        self.horizon = int(update_cfg.get("horizon", 6))
        self.eta = float(update_cfg.get("eta", 1.0))
        self.lyapunov_v0 = float(update_cfg.get("lyapunov_v0", 0.001))
        self.lyapunov_power = float(update_cfg.get("lyapunov_power", 0.5))
        self.update_strategy = str(update_cfg.get("strategy", "mpc_dp"))
        self.delta_threshold = float(update_cfg.get("delta_threshold", 5.0))
        self.weights = np.asarray(
            [update_cfg.get("cpu_weight", 0.5), update_cfg.get("memory_weight", 0.5)],
            dtype=np.float64,
        )
        if np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("CPU and memory weights must be positive and sum to one")
        self.max_hops = int(search_cfg.get("max_hops", 5))
        self.deadline_ms = float(search_cfg.get("deadline_ms", 180.0))
        self.top_k = int(search_cfg.get("top_k_confirmation", 3))
        self.max_cache_age_s = float(search_cfg.get("max_state_age_seconds", 10.0))
        self.search_strategy = str(search_cfg.get("strategy", "workflow"))
        self.gossip_fanout = int(search_cfg.get("gossip_fanout", 1))
        self.gamma = float(trust_cfg.get("path_decay_gamma", 0.9))
        self.trust_enabled = bool(trust_cfg.get("enabled", True))
        self.adversarial_overstate = bool(
            self.config.get("experiment", {}).get("adversarial_overstate", False)
        )
        self._validate_runtime_strategies()

        self.forecaster = FITSForecaster(model_path)
        self.history: list[np.ndarray] = []
        self.current_utilization = np.zeros(2, dtype=np.float64)
        self.current_forecast = np.zeros((self.horizon, 2), dtype=np.float64)
        self.last_inference_ms = math.nan
        self.last_sent: np.ndarray | None = None
        self.sequence = 0
        self.budget_guard = TokenBudget(self.budget, epoch_slots=self.horizon)
        self.lyapunov_queue = 0.0
        self.pause_at_budget_boundary = False
        self.monitor_paused = False
        self.cache: dict[str, CachedState] = {}
        self.trust = {node: [1.0, 1.0] for node in self.neighbors}
        self.seen_requests: dict[str, float] = {}
        self.late_candidates: dict[str, dict[str, Any]] = {}
        self.pending_tasks: set[asyncio.Task] = set()
        self.session: aiohttp.ClientSession | None = None
        self.monitor_task: asyncio.Task | None = None
        self.started_ms = now_ms()
        self.stats = {
            "monitor_samples": 0,
            "update_decisions": 0,
            "update_messages_attempted": 0,
            "update_messages_succeeded": 0,
            "searches_originated": 0,
            "search_forwards_received": 0,
            "search_forwards_sent": 0,
            "confirmations_attempted": 0,
            "confirmations_succeeded": 0,
            "duplicate_requests": 0,
        }
        log_dir = config_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{self.node_id}.jsonl"
        psutil.cpu_percent(interval=None)

    def log(self, event: str, **fields: Any) -> None:
        row = {"timestamp_ms": now_ms(), "node_id": self.node_id, "event": event, **fields}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _sample_resource(self) -> np.ndarray:
        return clamp_resource([psutil.cpu_percent(interval=None), psutil.virtual_memory().percent])

    def _forecast(self, current: np.ndarray) -> np.ndarray:
        future, elapsed_ms = self.forecaster.predict(self.history)
        self.last_inference_ms = elapsed_ms
        forecast = np.concatenate([current[None, :], future], axis=0)
        if len(forecast) < self.horizon:
            forecast = np.concatenate(
                [forecast, np.repeat(forecast[-1:], self.horizon - len(forecast), axis=0)],
                axis=0,
            )
        return forecast[: self.horizon]

    def _validate_runtime_strategies(self) -> None:
        if self.update_strategy not in {"mpc_dp", "periodic", "send_on_delta", "lyapunov"}:
            raise ValueError(f"unsupported update strategy: {self.update_strategy}")
        if self.search_strategy not in {"workflow", "gossip"}:
            raise ValueError(f"unsupported search strategy: {self.search_strategy}")
        if self.delta_threshold < 0:
            raise ValueError("delta_threshold must be non-negative")
        if self.gossip_fanout < 1:
            raise ValueError("gossip_fanout must be positive")

    def _propose_update(self, current: np.ndarray, forecast: np.ndarray) -> bool:
        return runtime_update_proposal(
            strategy=self.update_strategy,
            current=current,
            forecast=forecast,
            last_sent=self.last_sent,
            budget=self.budget,
            epoch_position=self.budget_guard.epoch_position,
            delta_threshold=getattr(self, "delta_threshold", 0.0),
            eta=getattr(self, "eta", 0.0),
            weights=self.weights,
            lyapunov_queue=getattr(self, "lyapunov_queue", 0.0),
            lyapunov_v0=getattr(self, "lyapunov_v0", 0.001),
            lyapunov_power=getattr(self, "lyapunov_power", 0.5),
            step_index=self.budget_guard.steps,
        )

    async def _broadcast_state(self) -> None:
        if self.session is None:
            return
        advertised_utilization = (
            np.zeros_like(self.current_utilization)
            if self.adversarial_overstate
            else self.current_utilization
        )
        advertised_forecast = (
            np.zeros_like(self.current_forecast)
            if self.adversarial_overstate
            else self.current_forecast
        )
        payload = {
            "node_id": self.node_id,
            "base_url": self.advertise_url,
            "timestamp_ms": now_ms(),
            "sequence": self.sequence,
            "utilization": dict(zip(RESOURCE_KEYS, advertised_utilization.tolist())),
            "availability": dict(zip(RESOURCE_KEYS, (100.0 - advertised_utilization).tolist())),
            "forecast_utilization": advertised_forecast.tolist(),
        }

        async def send(base_url: str) -> bool:
            self.stats["update_messages_attempted"] += 1
            try:
                timeout = aiohttp.ClientTimeout(total=max(0.2, self.interval * 0.8))
                async with self.session.post(base_url + "/state/update", json=payload, timeout=timeout) as response:
                    await response.read()
                    return response.status == 200
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return False

        results = await asyncio.gather(*(send(url) for url in self.neighbors.values()))
        succeeded = int(sum(results))
        self.stats["update_messages_succeeded"] += succeeded
        self.log(
            "state_broadcast",
            sequence=self.sequence,
            neighbors=len(results),
            succeeded=succeeded,
            utilization=self.current_utilization.tolist(),
            advertised_utilization=advertised_utilization.tolist(),
            adversarial_overstate=self.adversarial_overstate,
            inference_ms=self.last_inference_ms,
        )

    async def monitor_loop(self) -> None:
        while True:
            if (
                self.pause_at_budget_boundary
                and self.budget_guard.epoch_position == self.budget_guard.epoch_slots
            ):
                self.pause_at_budget_boundary = False
                self.monitor_paused = True
            if self.monitor_paused:
                await asyncio.sleep(0.05)
                continue
            started = time.monotonic()
            try:
                current = self._sample_resource()
                self.history.append(current.copy())
                self.history = self.history[-self.forecaster.seq_len :]
                self.current_utilization = current
                self.current_forecast = self._forecast(current)
                self.stats["monitor_samples"] += 1
                proposed = self._propose_update(current, self.current_forecast)
                action = self.budget_guard.allow(proposed)
                if self.update_strategy == "lyapunov":
                    self.lyapunov_queue = max(
                        self.lyapunov_queue + float(action) - self.budget, 0.0
                    )
                if action:
                    self.stats["update_decisions"] += 1
                    self.sequence += 1
                    self.last_sent = current.copy()
                    await self._broadcast_state()
                if (
                    self.pause_at_budget_boundary
                    and self.budget_guard.epoch_position == self.budget_guard.epoch_slots
                ):
                    self.pause_at_budget_boundary = False
                    self.monitor_paused = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the node alive and preserve evidence
                self.log("monitor_error", error=f"{type(exc).__name__}: {exc}")
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.01, self.interval - elapsed))

    def cache_is_fresh(self, state: CachedState) -> bool:
        return now_ms() - state.received_ms <= self.max_cache_age_s * 1000.0

    def _edge_trust(self, neighbor: str) -> float:
        if not self.trust_enabled:
            return 1.0
        alpha, beta = self.trust[neighbor]
        return direct_trust(alpha, beta)

    def _extend_path_trust(self, previous: float | None, edge_trust: float) -> float:
        if not self.trust_enabled:
            return 1.0
        return extend_path_trust(previous, edge_trust, self.gamma)

    def _forward_neighbors(
        self,
        path: list[str],
        request_id: str,
        strategy: str,
        gossip_fanout: int,
    ) -> list[str]:
        available = [neighbor for neighbor in self.neighbors if neighbor not in path]
        if strategy == "workflow":
            return available
        # Stable request-specific ordering gives reproducible gossip trials
        # without requiring synchronized random-number-generator state.
        ordered = sorted(
            available,
            key=lambda neighbor: hashlib.sha256(
                f"{request_id}|{self.node_id}|{neighbor}".encode("utf-8")
            ).digest(),
        )
        return ordered[: max(1, gossip_fanout)]

    def _candidate_from_neighbor(
        self,
        neighbor: str,
        demand: np.ndarray,
        path: list[str],
        prior_trust: float | None,
    ) -> dict[str, Any] | None:
        state = self.cache.get(neighbor)
        if state is None or not self.cache_is_fresh(state) or not qualifies(state.availability, demand):
            return None
        trust_value = self._extend_path_trust(prior_trust, self._edge_trust(neighbor))
        return {
            "node_id": neighbor,
            "base_url": state.base_url or self.neighbors[neighbor],
            "path": path + [neighbor],
            "path_trust": trust_value,
            "cached_availability": dict(zip(RESOURCE_KEYS, state.availability.tolist())),
            "cache_timestamp_ms": state.timestamp_ms,
            "trust_owner_id": self.node_id,
            "trust_owner_url": self.advertise_url,
        }

    def _deduplicate_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            node = str(candidate.get("node_id", ""))
            if not node or node == self.node_id:
                continue
            if node not in best or float(candidate["path_trust"]) > float(best[node]["path_trust"]):
                best[node] = candidate
        return list(best.values())

    async def _forward_request(
        self,
        neighbor: str,
        payload: dict[str, Any],
        deadline_unix_ms: float,
    ) -> list[dict[str, Any]]:
        if self.session is None:
            return []
        remaining = (deadline_unix_ms - now_ms()) / 1000.0
        if remaining <= 0:
            return []
        self.stats["search_forwards_sent"] += 1
        try:
            timeout = aiohttp.ClientTimeout(total=max(0.001, remaining))
            async with self.session.post(
                self.neighbors[neighbor] + "/search/forward",
                json=payload,
                timeout=timeout,
            ) as response:
                body = await response.json()
                if response.status != 200:
                    return []
                return list(body.get("candidates", []))
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return []

    async def _collect_forward_tasks(
        self,
        tasks: list[asyncio.Task],
        request_id: str,
        deadline_unix_ms: float,
    ) -> list[dict[str, Any]]:
        if not tasks:
            return []
        remaining = max(0.0, (deadline_unix_ms - now_ms()) / 1000.0)
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        candidates: list[dict[str, Any]] = []
        for task in done:
            try:
                candidates.extend(task.result())
            except Exception:
                pass
        for task in pending:
            self.pending_tasks.add(task)

            def finished(item: asyncio.Task, rid: str = request_id) -> None:
                self.pending_tasks.discard(item)
                try:
                    late = self._deduplicate_candidates(item.result())
                except Exception:
                    late = []
                if late:
                    self.late_candidates[rid] = {
                        "received_ms": now_ms(),
                        "candidates": late,
                    }
                    self.log("late_candidates_cached", request_id=rid, count=len(late))

            task.add_done_callback(finished)
        return candidates

    async def _confirm_candidate(
        self,
        candidate: dict[str, Any],
        demand: np.ndarray,
        request_id: str,
        deadline_unix_ms: float,
    ) -> tuple[dict[str, Any], bool, dict[str, Any] | None]:
        if self.session is None:
            return candidate, False, None
        remaining = (deadline_unix_ms - now_ms()) / 1000.0
        if remaining <= 0:
            return candidate, False, None
        self.stats["confirmations_attempted"] += 1
        try:
            timeout = aiohttp.ClientTimeout(total=max(0.001, remaining))
            candidate_node = str(candidate.get("node_id", ""))
            candidate_url = self.neighbors.get(
                candidate_node, str(candidate["base_url"]).rstrip("/")
            )
            async with self.session.post(
                candidate_url + "/confirm",
                json={
                    "request_id": request_id,
                    "demand": dict(zip(RESOURCE_KEYS, demand.tolist())),
                },
                timeout=timeout,
            ) as response:
                body = await response.json()
                valid = response.status == 200 and bool(body.get("qualifies", False))
                if valid:
                    self.stats["confirmations_succeeded"] += 1
                return candidate, valid, body
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return candidate, False, None

    async def _feedback(self, candidate: dict[str, Any], success: bool) -> None:
        target = str(candidate["node_id"])
        owner_id = str(candidate.get("trust_owner_id", ""))
        if owner_id == self.node_id and target in self.trust:
            self.trust[target][0 if success else 1] += 1.0
            return
        if self.session is None:
            return
        owner_url = self.neighbors.get(
            owner_id, str(candidate.get("trust_owner_url", "")).rstrip("/")
        )
        if not owner_url:
            return
        try:
            async with self.session.post(
                owner_url + "/trust/feedback",
                json={"target_node": target, "success": success},
                timeout=aiohttp.ClientTimeout(total=0.3),
            ) as response:
                await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

    async def search(self, body: dict[str, Any]) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        request_id = str(body.get("request_id") or uuid.uuid4())
        demand_body = body.get("demand", {})
        demand = clamp_resource([demand_body.get("cpu", 0.0), demand_body.get("memory", 0.0)])
        deadline_ms = float(body.get("deadline_ms", self.deadline_ms))
        max_hops = int(body.get("max_hops", self.max_hops))
        top_k = int(body.get("top_k", self.top_k))
        search_strategy = str(body.get("search_strategy", self.search_strategy))
        gossip_fanout = int(body.get("gossip_fanout", self.gossip_fanout))
        if search_strategy not in {"workflow", "gossip"}:
            raise ValueError(f"unsupported search strategy: {search_strategy}")
        deadline_unix_ms = now_ms() + deadline_ms
        self.stats["searches_originated"] += 1

        candidates = [
            item
            for neighbor in self.neighbors
            if (item := self._candidate_from_neighbor(neighbor, demand, [self.node_id], None))
            is not None
        ]
        mode = "direct-neighbor"
        if not candidates and max_hops > 1:
            mode = "multi-hop"
            tasks: list[asyncio.Task] = []
            for neighbor in self._forward_neighbors(
                [self.node_id], request_id, search_strategy, gossip_fanout
            ):
                edge_trust = self._edge_trust(neighbor)
                payload = {
                    "request_id": request_id,
                    "origin_id": self.node_id,
                    "path": [self.node_id, neighbor],
                    "path_trust": edge_trust,
                    "demand": dict(zip(RESOURCE_KEYS, demand.tolist())),
                    "deadline_unix_ms": deadline_unix_ms,
                    "max_hops": max_hops,
                    "search_strategy": search_strategy,
                    "gossip_fanout": gossip_fanout,
                }
                tasks.append(asyncio.create_task(self._forward_request(neighbor, payload, deadline_unix_ms)))
            candidates = await self._collect_forward_tasks(tasks, request_id, deadline_unix_ms)

        candidates = self._deduplicate_candidates(candidates)

        def rank_key(item: dict[str, Any]) -> tuple[float, float, int]:
            available = np.asarray(
                [item["cached_availability"]["cpu"], item["cached_availability"]["memory"]]
            )
            margin = float(np.min(available - demand))
            return (-float(item["path_trust"]), -margin, len(item.get("path", [])))

        ranked = sorted(candidates, key=rank_key)
        confirm_tasks = [
            asyncio.create_task(self._confirm_candidate(item, demand, request_id, deadline_unix_ms))
            for item in ranked[:top_k]
        ]
        confirmations: dict[str, tuple[bool, dict[str, Any] | None]] = {}
        if confirm_tasks:
            remaining = max(0.0, (deadline_unix_ms - now_ms()) / 1000.0)
            done, pending = await asyncio.wait(confirm_tasks, timeout=remaining)
            for task in pending:
                task.cancel()
            for task in done:
                try:
                    candidate, valid, confirmation = task.result()
                    confirmations[str(candidate["node_id"])] = (valid, confirmation)
                except Exception:
                    pass

        selected: dict[str, Any] | None = None
        for candidate in ranked[:top_k]:
            valid, confirmation = confirmations.get(str(candidate["node_id"]), (False, None))
            candidate["confirmed"] = valid
            candidate["confirmation"] = confirmation
            if selected is None and valid:
                selected = candidate

            # Trust feedback updates future decisions, but it is not part of the
            # current candidate decision.  Keep it off the response critical path.
            feedback_task = asyncio.create_task(self._feedback(candidate, valid))
            self.pending_tasks.add(feedback_task)

            def feedback_finished(item: asyncio.Task) -> None:
                self.pending_tasks.discard(item)
                try:
                    item.result()
                except Exception as exc:
                    self.log("trust_feedback_error", error=f"{type(exc).__name__}: {exc}")

            feedback_task.add_done_callback(feedback_finished)

        latency_ms = (time.perf_counter_ns() - started_ns) / 1e6
        result = {
            "request_id": request_id,
            "origin": self.node_id,
            "mode": mode,
            "search_strategy": search_strategy,
            "gossip_fanout": gossip_fanout if search_strategy == "gossip" else None,
            "success": selected is not None,
            "decision_latency_ms": latency_ms,
            "deadline_ms": deadline_ms,
            "demand": dict(zip(RESOURCE_KEYS, demand.tolist())),
            "candidate_count": len(ranked),
            "confirmed_count": sum(int(value[0]) for value in confirmations.values()),
            "selected": selected,
            "ranked_candidates": ranked[:top_k],
            "late_results_policy": "timestamped-cache-only; never changes this decision",
        }
        self.log(
            "search_complete",
            request_id=request_id,
            success=result["success"],
            mode=mode,
            latency_ms=latency_ms,
            candidate_count=len(ranked),
            selected=None if selected is None else selected["node_id"],
        )
        return result

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "node_id": self.node_id,
                "base_url": self.advertise_url,
                "uptime_seconds": (now_ms() - self.started_ms) / 1000.0,
                "utilization": dict(zip(RESOURCE_KEYS, self.current_utilization.tolist())),
                "availability": dict(zip(RESOURCE_KEYS, (100.0 - self.current_utilization).tolist())),
                "fits_inference_ms": self.last_inference_ms,
                "prediction_inference_ms": self.last_inference_ms,
                "prediction_model": self.forecaster.model_label,
                "prediction_model_key": self.forecaster.model_key,
                "prediction_checkpoint_sha256": self.forecaster.checkpoint_sha256,
                "history_samples": len(self.history),
                "cached_neighbors": len(self.cache),
                "configured_neighbors": list(self.neighbors),
                "update_budget_observed": (
                    self.budget_guard.used / self.budget_guard.steps if self.budget_guard.steps else 0.0
                ),
                "update_strategy": self.update_strategy,
                "search_strategy": self.search_strategy,
                "delta_threshold": self.delta_threshold,
                "lyapunov_v0": self.lyapunov_v0,
                "lyapunov_queue": self.lyapunov_queue,
                "budget_epoch_position": self.budget_guard.epoch_position,
                "budget_epoch_slots": self.budget_guard.epoch_slots,
                "monitor_paused": self.monitor_paused,
                "gossip_fanout": self.gossip_fanout,
                "trust_enabled": self.trust_enabled,
                "adversarial_overstate": self.adversarial_overstate,
                "stats": self.stats,
            }
        )

    async def handle_state_update(self, request: web.Request) -> web.Response:
        body = await request.json()
        sender = str(body.get("node_id", ""))
        if sender not in self.neighbors:
            raise web.HTTPForbidden(text="state updates are accepted only from business neighbors")
        sequence = int(body.get("sequence", -1))
        previous = self.cache.get(sender)
        if previous is not None and sequence <= previous.sequence:
            return web.json_response({"accepted": False, "reason": "stale-sequence"})
        utilization = clamp_resource(
            [body["utilization"]["cpu"], body["utilization"]["memory"]]
        )
        availability = clamp_resource(
            [body["availability"]["cpu"], body["availability"]["memory"]]
        )
        state = CachedState(
            node_id=sender,
            # The receiver's configured address is authoritative for a direct
            # business neighbor.  This permits subnet-local routing while the
            # advertised URL remains globally reachable for indirect results.
            base_url=str(self.neighbors[sender]).rstrip("/"),
            timestamp_ms=float(body["timestamp_ms"]),
            received_ms=now_ms(),
            sequence=sequence,
            utilization=utilization,
            availability=availability,
            forecast_utilization=np.asarray(body.get("forecast_utilization", []), dtype=np.float64),
        )
        self.cache[sender] = state
        return web.json_response({"accepted": True, "node_id": self.node_id})

    async def handle_confirm(self, request: web.Request) -> web.Response:
        body = await request.json()
        demand_body = body.get("demand", {})
        demand = clamp_resource([demand_body.get("cpu", 0.0), demand_body.get("memory", 0.0)])
        available = 100.0 - self.current_utilization
        return web.json_response(
            {
                "node_id": self.node_id,
                "base_url": self.advertise_url,
                "timestamp_ms": now_ms(),
                "alive": True,
                "availability": dict(zip(RESOURCE_KEYS, available.tolist())),
                "qualifies": qualifies(available, demand),
            }
        )

    async def handle_search(self, request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response(await self.search(body))

    async def handle_forward(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.stats["search_forwards_received"] += 1
        request_id = str(body["request_id"])
        path = [str(item) for item in body["path"]]
        if not path or path[-1] != self.node_id:
            raise web.HTTPBadRequest(text="forward path must end at the receiving node")
        if len(path) < 2 or path[-2] not in self.neighbors:
            raise web.HTTPForbidden(text="search forwards are accepted only from business neighbors")
        deadline_unix_ms = float(body["deadline_unix_ms"])
        if now_ms() > deadline_unix_ms:
            return web.json_response({"request_id": request_id, "candidates": [], "expired": True})

        expiry = self.seen_requests.get(request_id)
        if expiry is not None and expiry >= now_ms():
            self.stats["duplicate_requests"] += 1
            return web.json_response({"request_id": request_id, "candidates": [], "duplicate": True})
        self.seen_requests[request_id] = deadline_unix_ms + 60_000.0
        cutoff = now_ms() - 60_000.0
        self.seen_requests = {key: value for key, value in self.seen_requests.items() if value >= cutoff}

        demand_body = body.get("demand", {})
        demand = clamp_resource([demand_body.get("cpu", 0.0), demand_body.get("memory", 0.0)])
        previous_trust = float(body["path_trust"])
        max_hops = int(body.get("max_hops", self.max_hops))
        search_strategy = str(body.get("search_strategy", self.search_strategy))
        gossip_fanout = int(body.get("gossip_fanout", self.gossip_fanout))
        if not can_reach_one_more_hop(path, max_hops):
            return web.json_response({"request_id": request_id, "candidates": []})
        candidates = [
            item
            for neighbor in self.neighbors
            if neighbor not in path
            and (
                item := self._candidate_from_neighbor(
                    neighbor,
                    demand,
                    path,
                    previous_trust,
                )
            )
            is not None
        ]
        if candidates:
            return web.json_response({"request_id": request_id, "candidates": candidates})

        tasks: list[asyncio.Task] = []
        for neighbor in self._forward_neighbors(
            path, request_id, search_strategy, gossip_fanout
        ):
            edge_trust = self._extend_path_trust(previous_trust, self._edge_trust(neighbor))
            forwarded = dict(body)
            forwarded["path"] = path + [neighbor]
            forwarded["path_trust"] = edge_trust
            tasks.append(
                asyncio.create_task(self._forward_request(neighbor, forwarded, deadline_unix_ms))
            )
        children = await self._collect_forward_tasks(tasks, request_id, deadline_unix_ms)
        return web.json_response(
            {"request_id": request_id, "candidates": self._deduplicate_candidates(children)}
        )

    async def handle_feedback(self, request: web.Request) -> web.Response:
        body = await request.json()
        target = str(body.get("target_node", ""))
        if target not in self.trust:
            raise web.HTTPBadRequest(text="target is not a direct business neighbor")
        success = bool(body.get("success", False))
        self.trust[target][0 if success else 1] += 1.0
        return web.json_response(
            {
                "updated": True,
                "target_node": target,
                "trust": self._edge_trust(target),
                "alpha": self.trust[target][0],
                "beta": self.trust[target][1],
            }
        )

    async def handle_experiment_config(self, request: web.Request) -> web.Response:
        """Switch an in-memory experiment mode on the trusted test LAN."""
        body = await request.json()
        if "update_strategy" in body:
            self.update_strategy = str(body["update_strategy"])
        if "search_strategy" in body:
            self.search_strategy = str(body["search_strategy"])
        if "delta_threshold" in body:
            self.delta_threshold = float(body["delta_threshold"])
        if "lyapunov_v0" in body:
            self.lyapunov_v0 = float(body["lyapunov_v0"])
        if "gossip_fanout" in body:
            self.gossip_fanout = int(body["gossip_fanout"])
        if "trust_enabled" in body:
            self.trust_enabled = bool(body["trust_enabled"])
        if "adversarial_overstate" in body:
            self.adversarial_overstate = bool(body["adversarial_overstate"])
        if "pause_at_budget_boundary" in body:
            self.pause_at_budget_boundary = bool(body["pause_at_budget_boundary"])
            if not self.pause_at_budget_boundary:
                self.monitor_paused = False
        self._validate_runtime_strategies()
        if bool(body.get("reset", False)):
            self.cache.clear()
            self.trust = {node: [1.0, 1.0] for node in self.neighbors}
            self.seen_requests.clear()
            self.late_candidates.clear()
            # Each paired method must rebuild the predictor context from the
            # same workload warm-up instead of inheriting samples produced by
            # the method that ran immediately before it.
            self.history.clear()
            self.current_utilization = np.zeros(2, dtype=np.float64)
            self.current_forecast = np.zeros((self.horizon, 2), dtype=np.float64)
            self.last_inference_ms = math.nan
            self.last_sent = None
            self.sequence = 0
            self.budget_guard = TokenBudget(self.budget, epoch_slots=self.horizon)
            self.lyapunov_queue = 0.0
            self.pause_at_budget_boundary = False
            self.monitor_paused = False
            for key in self.stats:
                self.stats[key] = 0
        elif bool(body.get("measurement_reset", False)):
            # Start a fair measurement window without destroying the cache and
            # predictor history built during the common warm-up.  Resetting the
            # repeated budget epoch here prevents policy-dependent warm-up
            # duration from changing the update allowance seen by requests.
            self.seen_requests.clear()
            self.late_candidates.clear()
            self.budget_guard = TokenBudget(self.budget, epoch_slots=self.horizon)
            self.lyapunov_queue = 0.0
            self.pause_at_budget_boundary = False
            self.monitor_paused = False
            for key in self.stats:
                self.stats[key] = 0
        return web.json_response(
            {
                "node_id": self.node_id,
                "update_strategy": self.update_strategy,
                "search_strategy": self.search_strategy,
                "delta_threshold": self.delta_threshold,
                "lyapunov_v0": self.lyapunov_v0,
                "lyapunov_queue": self.lyapunov_queue,
                "gossip_fanout": self.gossip_fanout,
                "trust_enabled": self.trust_enabled,
                "adversarial_overstate": self.adversarial_overstate,
                "reset": bool(body.get("reset", False)),
                "measurement_reset": bool(body.get("measurement_reset", False)),
                "pause_at_budget_boundary": self.pause_at_budget_boundary,
                "monitor_paused": self.monitor_paused,
            }
        )

    async def handle_debug_state(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "node_id": self.node_id,
                "cache": {node: state.public() for node, state in self.cache.items()},
                "trust": {
                    node: {
                        "alpha": values[0],
                        "beta": values[1],
                        "value": direct_trust(*values),
                    }
                    for node, values in self.trust.items()
                },
                "late_candidates": self.late_candidates,
                "trust_enabled": self.trust_enabled,
                "adversarial_overstate": self.adversarial_overstate,
                "stats": self.stats,
            }
        )

    async def startup(self, app: web.Application) -> None:
        connector = aiohttp.TCPConnector(limit=64, ttl_dns_cache=30)
        self.session = aiohttp.ClientSession(connector=connector)
        self.monitor_task = asyncio.create_task(self.monitor_loop())
        self.log("agent_started", advertise_url=self.advertise_url, neighbors=self.neighbors)

    async def cleanup(self, app: web.Application) -> None:
        if self.monitor_task is not None:
            self.monitor_task.cancel()
            await asyncio.gather(self.monitor_task, return_exceptions=True)
        for task in self.pending_tasks:
            task.cancel()
        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks, return_exceptions=True)
        if self.session is not None:
            await self.session.close()
        self.log("agent_stopped")

    def create_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024)
        app.add_routes(
            [
                web.get("/health", self.handle_health),
                web.post("/state/update", self.handle_state_update),
                web.post("/search", self.handle_search),
                web.post("/search/forward", self.handle_forward),
                web.post("/confirm", self.handle_confirm),
                web.post("/trust/feedback", self.handle_feedback),
                web.post("/experiment/config", self.handle_experiment_config),
                web.get("/debug/state", self.handle_debug_state),
            ]
        )
        app.on_startup.append(self.startup)
        app.on_cleanup.append(self.cleanup)
        return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed edge-node discovery agent")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = NodeAgent(args.config.resolve(), args.model.resolve())
    web.run_app(agent.create_app(), host=agent.host, port=agent.port, print=None)


if __name__ == "__main__":
    main()
