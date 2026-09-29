from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


RAFT_ROOT = Path(__file__).resolve().parents[1]
if str(RAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(RAFT_ROOT))

from edgebench_core import (  # noqa: E402
    ExactTraceBudgetGuard,
    RepeatedEpochBudgetGuard,
    mpc_dp_first_action,
    runtime_update_proposal,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NODE_MODULE = load_module("conformance_node_agent", Path(__file__).with_name("node_agent.py"))
SIM_MODULE = load_module(
    "conformance_update_experiment",
    RAFT_ROOT / "update_experiments" / "run_update_experiments.py",
)


class SharedUpdatePolicyConformanceTests(unittest.TestCase):
    def test_both_adapters_import_the_shared_mpc_function(self) -> None:
        self.assertIs(NODE_MODULE.mpc_dp_first_action, mpc_dp_first_action)
        self.assertIs(SIM_MODULE.mpc_dp_first_action, mpc_dp_first_action)
        self.assertIs(NODE_MODULE.TokenBudget, RepeatedEpochBudgetGuard)
        self.assertIs(SIM_MODULE.BudgetGuard, ExactTraceBudgetGuard)

    def test_node_adapter_and_shared_runtime_match_slot_by_slot(self) -> None:
        slots = np.arange(24, dtype=float)
        actual = np.column_stack(
            [35.0 + 18.0 * np.sin(slots / 2.7), 45.0 + 12.0 * np.cos(slots / 3.1)]
        )
        horizon = 6
        forecasts = np.empty((len(actual), horizon, 2), dtype=float)
        for t in range(len(actual)):
            for h in range(horizon):
                forecasts[t, h] = actual[min(t + h, len(actual) - 1)]

        for strategy in ("mpc_dp", "periodic", "send_on_delta"):
            with self.subTest(strategy=strategy):
                agent = NODE_MODULE.NodeAgent.__new__(NODE_MODULE.NodeAgent)
                agent.update_strategy = strategy
                agent.budget = 0.5
                agent.delta_threshold = 5.0
                agent.eta = 1.0
                agent.weights = np.array([0.5, 0.5])
                agent.last_sent = None
                agent.budget_guard = NODE_MODULE.TokenBudget(0.5, epoch_slots=horizon)

                guard = RepeatedEpochBudgetGuard(0.5, epoch_slots=horizon)
                shared_last_sent = None
                for t, current in enumerate(actual):
                    agent_proposal = agent._propose_update(current, forecasts[t])
                    shared_proposal = runtime_update_proposal(
                        strategy=strategy,
                        current=current,
                        forecast=forecasts[t],
                        last_sent=shared_last_sent,
                        budget=0.5,
                        epoch_position=guard.epoch_position,
                        delta_threshold=5.0,
                        eta=1.0,
                        weights=np.array([0.5, 0.5]),
                    )
                    self.assertEqual(agent_proposal, shared_proposal)
                    agent_action = agent.budget_guard.allow(agent_proposal)
                    shared_action = guard.allow(shared_proposal)
                    self.assertEqual(agent_action, shared_action)
                    if agent_action:
                        agent.last_sent = current.copy()
                    if shared_action:
                        shared_last_sent = current.copy()
                self.assertEqual(agent.budget_guard.used, 12)
                self.assertEqual(guard.used, 12)


if __name__ == "__main__":
    unittest.main()
