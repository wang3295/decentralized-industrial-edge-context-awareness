import numpy as np

from run_weight_eta_task_sensitivity import (
    PROFILES,
    REFERENCE_LEVELS,
    profile_closeness,
    scenario_grid,
)


def run_self_checks() -> None:
    scenarios = scenario_grid()
    assert len(scenarios) == 10
    assert sum(factor == "cpu_weight" for factor, *_ in scenarios) == 5
    assert sum(factor == "eta" for factor, *_ in scenarios) == 5
    assert ("cpu_weight", REFERENCE_LEVELS["cpu_weight"], 0.5, 0.5) in scenarios
    assert ("eta", REFERENCE_LEVELS["eta"], 0.5, 0.5) in scenarios
    rng = np.random.default_rng(1)
    balanced = profile_closeness("balanced", rng)
    cpu = profile_closeness("cpu_intensive", rng)
    memory = profile_closeness("memory_intensive", rng)
    assert balanced[0] > 0.89 and balanced[1] > 0.89
    assert cpu[0] > cpu[1]
    assert memory[1] > memory[0]
    assert set(PROFILES) == {"balanced", "cpu_intensive", "memory_intensive"}
    print("10 weight/eta/task-profile self-checks passed")


if __name__ == "__main__":
    run_self_checks()
