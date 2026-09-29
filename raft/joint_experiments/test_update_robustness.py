from pathlib import Path

import numpy as np

from run_joint_discovery_experiment import BudgetGuard, load_inputs
from run_update_robustness_experiment import (
    calibrate_delta_threshold,
    holm_adjust,
    replay_delta_for_calibration,
    scenario_grid,
)


def run_self_checks() -> None:
    scenarios = scenario_grid()
    assert len(scenarios) == 15
    assert sum(factor == "update_budget" for factor, *_ in scenarios) == 5
    assert sum(factor == "outage_rate" for factor, *_ in scenarios) == 5
    assert sum(factor == "topology_churn_rate" for factor, *_ in scenarios) == 5

    adjusted = holm_adjust(np.array([0.01, 0.04, 0.20]))
    assert np.allclose(adjusted, np.array([0.03, 0.08, 0.20]))
    assert np.all((0.0 <= adjusted) & (adjusted <= 1.0))

    actual, _ = load_inputs(Path(__file__).resolve().parents[1] / "test")
    thresholds = [calibrate_delta_threshold(actual, 94, budget, 6) for budget in (0.25, 0.5, 0.75)]
    assert np.all(np.isfinite(thresholds))
    assert np.all(np.asarray(thresholds) >= 0.0)

    validation = actual[47:88]
    cache = replay_delta_for_calibration(validation, actual[46], 0.5, thresholds[1])
    assert cache.shape == validation.shape
    guard = BudgetGuard(len(validation), 0.5)
    assert guard.target == 20
    print("12 update-robustness self-checks passed")


if __name__ == "__main__":
    run_self_checks()
