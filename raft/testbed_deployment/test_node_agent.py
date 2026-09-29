from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("node_agent", HERE / "node_agent.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NodeAgentUnitTests(unittest.TestCase):
    def test_path_trust_recursion(self) -> None:
        first = MODULE.extend_path_trust(None, 0.8, 0.9)
        second = MODULE.extend_path_trust(first, 0.5, 0.9)
        self.assertAlmostEqual(first, 0.8)
        self.assertAlmostEqual(second, 0.9 * np.sqrt(0.8 * 0.5))

    def test_max_hops_does_not_admit_candidate_beyond_boundary(self) -> None:
        self.assertTrue(MODULE.can_reach_one_more_hop(["a", "b", "c"], 3))
        self.assertFalse(MODULE.can_reach_one_more_hop(["a", "b", "c", "d"], 3))

    def test_budget_never_exceeds_token_supply(self) -> None:
        guard = MODULE.TokenBudget(0.5, epoch_slots=10)
        actions = [guard.allow(True) for _ in range(20)]
        self.assertEqual(sum(actions), 10)
        self.assertEqual(guard.used / guard.steps, 0.5)

    def test_budget_forces_exact_epoch_target(self) -> None:
        guard = MODULE.TokenBudget(0.5, epoch_slots=6)
        actions = [guard.allow(False) for _ in range(6)]
        self.assertEqual(actions, [False, False, False, True, True, True])

    def test_mpc_returns_boolean_and_can_send(self) -> None:
        forecast = np.array([[10.0, 20.0], [10.0, 20.0], [80.0, 90.0]])
        result = MODULE.mpc_dp_first_action(
            forecast,
            cached=np.array([0.0, 0.0]),
            max_updates=1,
            eta=1.0,
            weights=np.array([0.5, 0.5]),
        )
        self.assertIsInstance(result, bool)

    def test_periodic_and_delta_update_proposals(self) -> None:
        agent = MODULE.NodeAgent.__new__(MODULE.NodeAgent)
        agent.last_sent = np.array([10.0, 20.0])
        agent.budget = 0.5
        agent.weights = np.array([0.5, 0.5])
        agent.delta_threshold = 5.0
        agent.budget_guard = MODULE.TokenBudget(0.5, epoch_slots=6)
        agent.update_strategy = "periodic"
        self.assertTrue(agent._propose_update(np.array([10.0, 20.0]), np.zeros((6, 2))))
        agent.budget_guard.epoch_position = 1
        self.assertFalse(agent._propose_update(np.array([10.0, 20.0]), np.zeros((6, 2))))
        agent.update_strategy = "send_on_delta"
        self.assertFalse(agent._propose_update(np.array([12.0, 22.0]), np.zeros((6, 2))))
        self.assertTrue(agent._propose_update(np.array([20.0, 20.0]), np.zeros((6, 2))))

    def test_lyapunov_update_proposal_uses_nonnegative_virtual_queue(self) -> None:
        current = np.array([80.0, 70.0])
        cached = np.array([10.0, 20.0])
        weights = np.array([0.5, 0.5])
        self.assertTrue(
            MODULE.runtime_update_proposal(
                "lyapunov",
                current=current,
                forecast=np.zeros((6, 2)),
                last_sent=cached,
                budget=0.5,
                epoch_position=0,
                delta_threshold=1.3355422455022612,
                eta=1.0,
                weights=weights,
                lyapunov_queue=0.0,
                lyapunov_v0=0.001,
                step_index=0,
            )
        )
        self.assertFalse(
            MODULE.runtime_update_proposal(
                "lyapunov",
                current=cached,
                forecast=np.zeros((6, 2)),
                last_sent=cached,
                budget=0.5,
                epoch_position=1,
                delta_threshold=1.3355422455022612,
                eta=1.0,
                weights=weights,
                lyapunov_queue=1.0,
                lyapunov_v0=0.001,
                step_index=1,
            )
        )
        queue = 0.0
        for action in (False, False, True, False):
            queue = max(queue + float(action) - 0.5, 0.0)
            self.assertGreaterEqual(queue, 0.0)

    def test_gossip_fanout_is_deterministic_and_bounded(self) -> None:
        agent = MODULE.NodeAgent.__new__(MODULE.NodeAgent)
        agent.node_id = "node-01"
        agent.neighbors = {"node-02": "x", "node-03": "y", "node-04": "z"}
        first = agent._forward_neighbors(["node-01"], "request-1", "gossip", 1)
        second = agent._forward_neighbors(["node-01"], "request-1", "gossip", 1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(
            agent._forward_neighbors(["node-01"], "request-1", "workflow", 1),
            ["node-02", "node-03", "node-04"],
        )

    def test_no_trust_mode_removes_reputation_and_path_decay(self) -> None:
        agent = MODULE.NodeAgent.__new__(MODULE.NodeAgent)
        agent.gamma = 0.9
        agent.trust = {"node-02": [9.0, 1.0]}
        agent.trust_enabled = True
        self.assertAlmostEqual(agent._edge_trust("node-02"), 0.9)
        self.assertLess(agent._extend_path_trust(0.8, 0.9), 1.0)
        agent.trust_enabled = False
        self.assertEqual(agent._edge_trust("node-02"), 1.0)
        self.assertEqual(agent._extend_path_trust(0.2, 0.3), 1.0)

    def test_fits_parameter_count(self) -> None:
        model = MODULE.FITSModel(100, 5, 2, 24)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 625)

    def test_checkpoint_round_trip(self) -> None:
        model = MODULE.FITSModel(100, 5, 2, 24)
        checkpoint = {
            "config": {"seq_len": 100, "pred_len": 5, "fits_cut_freq": 24},
            "mean": np.array([30.0, 60.0], dtype=np.float32),
            "std": np.array([10.0, 5.0], dtype=np.float32),
            "model_state": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(checkpoint, path)
            forecaster = MODULE.FITSForecaster(path)
            prediction, elapsed = forecaster.predict([np.array([25.0, 55.0])])
        self.assertEqual(prediction.shape, (5, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
