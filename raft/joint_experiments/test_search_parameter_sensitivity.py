import numpy as np

from run_search_parameter_sensitivity import REFERENCE_LEVELS, holm_adjust, scenario_grid


def run_self_checks() -> None:
    scenarios = scenario_grid()
    assert len(scenarios) == 17
    assert sum(factor == "max_hops" for factor, _ in scenarios) == 6
    assert sum(factor == "deadline_ms" for factor, _ in scenarios) == 6
    assert sum(factor == "top_k" for factor, _ in scenarios) == 5
    for factor, reference in REFERENCE_LEVELS.items():
        assert (factor, reference) in scenarios
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.20]))
    assert np.allclose(adjusted, np.array([0.03, 0.08, 0.20]))
    print("10 search-parameter self-checks passed")


if __name__ == "__main__":
    run_self_checks()
