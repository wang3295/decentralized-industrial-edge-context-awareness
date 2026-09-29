import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_update_experiments.py")
SPEC = importlib.util.spec_from_file_location("update_experiment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_weighted_error_is_not_double_weighted():
    x = np.array([2.0, 4.0])
    z = np.array([0.0, 0.0])
    assert MODULE.weighted_squared_error(x, z) == 0.5 * 4.0 + 0.5 * 16.0
    assert MODULE.weighted_squared_error(x, z, np.array([0.8, 0.2])) == 0.8 * 4.0 + 0.2 * 16.0


def test_budget_guard_exact_and_prefix_safe():
    for proposed_value in (False, True):
        guard = MODULE.BudgetGuard(100, 0.3)
        actions = [guard.apply(proposed_value, t) for t in range(100)]
        assert sum(actions) == 30
        assert guard.violations == 0


def test_all_methods_have_identical_budget():
    length = 40
    actual = np.column_stack([np.arange(length), np.arange(length) * 0.5])
    predictions = np.repeat(actual[:, None, :], 3, axis=1)
    for t in range(length):
        for h in range(3):
            predictions[t, h] = actual[min(t + h, length - 1)]
    params = {
        "Random": None,
        "Periodic": None,
        "Send-on-delta": 1.0,
        "Lyapunov": 0.05,
        "MPC-DP": 0.2,
    }
    for method, parameter in params.items():
        result = MODULE.run_strategy(
            method,
            actual,
            predictions,
            initial_cache=np.array([-1.0, -0.5]),
            budget=0.25,
            parameter=parameter,
            horizon=3,
        )
        assert result.update_count == 10
        assert result.budget_violation_count == 0


def test_mpc_uses_current_update_for_large_error():
    forecast = np.array([[10.0, 10.0], [10.0, 10.0]])
    cached = np.array([0.0, 0.0])
    assert MODULE.mpc_dp_first_action(forecast, cached, max_updates=1, eta=0.1)


def test_mpc_accepts_an_explicit_three_resource_vector():
    weights = np.array([1.0 / 3.0] * 3)
    forecast = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])
    cached = np.array([0.0, 0.0, 0.0])
    assert MODULE.mpc_dp_first_action(
        forecast,
        cached,
        max_updates=1,
        eta=0.1,
        weights=weights,
    )


def test_oracle_and_last_value_prediction_construction():
    actual = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    oracle = MODULE.make_oracle_predictions(actual, 3)
    last = MODULE.make_last_value_predictions(actual, 3)
    np.testing.assert_array_equal(oracle[0], actual)
    np.testing.assert_array_equal(oracle[-1], np.repeat(actual[-1][None, :], 3, axis=0))
    np.testing.assert_array_equal(last[1], np.repeat(actual[1][None, :], 3, axis=0))


def test_lost_updates_consume_budget_but_do_not_refresh_cache():
    length = 20
    actual = np.column_stack([np.arange(length), np.arange(length)])
    predictions = np.repeat(actual[:, None, :], 2, axis=1)
    result = MODULE.run_strategy(
        "Periodic",
        actual,
        predictions,
        initial_cache=np.array([-1.0, -1.0]),
        budget=0.5,
        loss_probability=1.0,
    )
    assert result.update_count == 10
    assert result.successful_update_count == 0
    assert result.failed_update_count == 10
    assert result.effective_update_rate == 0.0
