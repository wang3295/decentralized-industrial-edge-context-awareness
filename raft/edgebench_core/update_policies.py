"""Shared resource-update decisions used by simulation and deployment code.

The pure MPC-DP objective lives here so that a simulator and a deployed node
cannot silently drift apart.  The two budget guards are deliberately named:
historical trace experiments used one exact budget over the complete trace,
whereas the online node agent used a repeated finite epoch.  New formal runs
must record which guard is used instead of treating the two semantics as the
same algorithm configuration.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


DEFAULT_WEIGHTS = np.asarray([0.5, 0.5], dtype=float)


def normalize_weights(weights: np.ndarray | None = None) -> np.ndarray:
    """Return validated positive resource weights that sum to one.

    Omitting ``weights`` preserves the two-resource CPU/memory default used by
    the main experiments.  Supplying a vector explicitly supports the general
    ``d``-dimensional formulation, including the three-resource throughput
    experiment, without changing the objective.
    """

    result = DEFAULT_WEIGHTS.copy() if weights is None else np.asarray(weights, dtype=float).copy()
    if result.ndim != 1 or result.size == 0 or np.any(result <= 0) or not np.isclose(result.sum(), 1.0):
        raise ValueError(f"weights must be a positive one-dimensional vector summing to one, got {result}")
    return result


def weighted_squared_error(
    x: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Resource-weighted squared error without applying weights twice."""

    active = normalize_weights(weights)
    return float(np.sum(active * np.square(np.asarray(x, dtype=float) - np.asarray(z, dtype=float))))


def mpc_dp_first_action(
    forecast: np.ndarray,
    cached: np.ndarray,
    max_updates: int,
    eta: float,
    weights: np.ndarray | None = None,
) -> bool:
    """Solve the paper's original finite-horizon DP and return its first action.

    The terminal state contains exactly ``max_updates`` transmissions.  A
    silent slot is charged the weighted cache error and, after at least one
    planned update, the Equation-(13) waiting penalty ``eta``.
    """

    values = np.asarray(forecast, dtype=float)
    cache = np.asarray(cached, dtype=float)
    active_weights = normalize_weights(weights)
    dimensions = int(active_weights.size)
    if values.ndim != 2 or values.shape[1] != dimensions:
        raise ValueError(
            f"forecast must have shape (horizon, {dimensions}), got {values.shape}"
        )
    if cache.shape != (dimensions,):
        raise ValueError(f"cached must have shape ({dimensions},), got {cache.shape}")
    length = int(values.shape[0])
    max_updates = min(max(int(max_updates), 0), length)

    @lru_cache(maxsize=None)
    def solve(h: int, used: int, last_index: int) -> tuple[float, tuple[int, ...]]:
        if h == length:
            return (0.0, ()) if used == max_updates else (float("inf"), ())
        cache_value = cache if last_index < 0 else values[last_index]
        slots_after_current = length - h - 1
        updates_needed = max_updates - used
        best = (float("inf"), ())
        if updates_needed <= slots_after_current:
            future_cost, plan = solve(h + 1, used, last_index)
            waiting_penalty = float(eta) if used > 0 else 0.0
            cost = weighted_squared_error(values[h], cache_value, active_weights)
            best = (cost + waiting_penalty + future_cost, (0,) + plan)
        if used < max_updates:
            future_cost, plan = solve(h + 1, used + 1, h)
            candidate = (future_cost, (1,) + plan)
            if candidate[0] < best[0] - 1e-12:
                best = candidate
        plan = best[1]
        return best[0], plan

    plan = solve(0, 0, -1)[1]
    return bool(plan and plan[0])


class ExactTraceBudgetGuard:
    """Exact budget with a prefix envelope over one finite trace."""

    def __init__(self, length: int, rate: float):
        if length <= 0:
            raise ValueError("length must be positive")
        if not 0.0 < rate < 1.0:
            raise ValueError("budget rate must be in (0, 1)")
        self.length = int(length)
        self.target = min(max(int(math.floor(rate * length + 1e-12)), 1), length - 1)
        self.used = 0
        self.blocked = 0
        self.violations = 0

    def apply(self, proposed: bool, t: int) -> bool:
        if not 0 <= int(t) < self.length:
            raise ValueError(f"t must be in [0, {self.length}), got {t}")
        remaining_slots = self.length - int(t)
        remaining_messages = self.target - self.used
        forced = remaining_messages >= remaining_slots
        prefix_limit = int(math.ceil(self.target * (int(t) + 1) / self.length))
        allowed = self.used < prefix_limit and self.used < self.target
        if proposed and not allowed:
            self.blocked += 1
        action = bool(forced or (proposed and allowed))
        if action:
            self.used += 1
        if self.used > prefix_limit or self.used > self.target:
            self.violations += 1
        return action


class RepeatedEpochBudgetGuard:
    """Exact budget repeated over fixed online epochs."""

    def __init__(self, budget: float, epoch_slots: int = 20):
        if not 0.0 < budget <= 1.0:
            raise ValueError("update budget must be in (0, 1]")
        if epoch_slots < 2:
            raise ValueError("epoch_slots must be at least two")
        self.budget = float(budget)
        self.epoch_slots = int(epoch_slots)
        self.epoch_target = min(
            max(int(math.floor(self.budget * self.epoch_slots + 1e-12)), 1),
            self.epoch_slots,
        )
        self.epoch_position = 0
        self.epoch_used = 0
        self.steps = 0
        self.used = 0

    @property
    def remaining_messages(self) -> int:
        return self.epoch_target - self.epoch_used

    def allow(self, proposed: bool) -> bool:
        if self.epoch_position == self.epoch_slots:
            if self.epoch_used != self.epoch_target:
                raise AssertionError("the repeated budget epoch did not finish at its exact target")
            self.epoch_position = 0
            self.epoch_used = 0

        remaining_slots = self.epoch_slots - self.epoch_position
        remaining_messages = self.epoch_target - self.epoch_used
        forced = remaining_messages >= remaining_slots
        prefix_limit = int(
            math.ceil(self.epoch_target * (self.epoch_position + 1) / self.epoch_slots)
        )
        allowed = self.epoch_used < prefix_limit and self.epoch_used < self.epoch_target
        action = bool(forced or (proposed and allowed))
        if action:
            self.epoch_used += 1
            self.used += 1
        self.epoch_position += 1
        self.steps += 1
        return action


def runtime_update_proposal(
    strategy: str,
    current: np.ndarray,
    forecast: np.ndarray,
    last_sent: np.ndarray | None,
    budget: float,
    epoch_position: int,
    delta_threshold: float,
    eta: float,
    weights: np.ndarray | None = None,
    lyapunov_queue: float = 0.0,
    lyapunov_v0: float = 0.001,
    lyapunov_power: float = 0.5,
    step_index: int = 0,
) -> bool:
    """Deployment-equivalent proposal used before the repeated budget guard.

    This preserves the semantics of the completed 16-node ``formal_v2`` run:
    periodic proposals start at epoch position zero and Send-on-delta uses the
    weighted L1 change.  Future algorithm versions should add a new explicitly
    named function instead of changing this compatibility function silently.
    """

    if last_sent is None:
        return True
    active_weights = normalize_weights(weights)
    current_value = np.asarray(current, dtype=float)
    last_value = np.asarray(last_sent, dtype=float)
    if strategy == "periodic":
        period = max(1, int(round(1.0 / float(budget))))
        return int(epoch_position) % period == 0
    if strategy == "send_on_delta":
        weighted_delta = float(np.sum(active_weights * np.abs(current_value - last_value)))
        return weighted_delta >= float(delta_threshold)
    if strategy == "lyapunov":
        error_if_silent = weighted_squared_error(current_value, last_value, active_weights)
        vt = float(lyapunov_v0) * (int(step_index) + 2) ** float(lyapunov_power)
        cost_silent = vt * error_if_silent + float(lyapunov_queue) * (0.0 - budget)
        cost_send = float(lyapunov_queue) * (1.0 - budget)
        return cost_send < cost_silent
    if strategy != "mpc_dp":
        raise ValueError(f"unsupported update strategy: {strategy}")
    values = np.asarray(forecast, dtype=float)
    max_updates = int(math.ceil(float(budget) * len(values)))
    return mpc_dp_first_action(
        values,
        last_value,
        max_updates=max_updates,
        eta=float(eta),
        weights=active_weights,
    )
