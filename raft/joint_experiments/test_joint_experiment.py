from pathlib import Path

import numpy as np

from run_joint_discovery_experiment import (
    JointConfig,
    build_cache_replays,
    build_node_replays,
    generate_tasks,
    graph_and_trust,
    load_inputs,
    run_joint_trials,
    run_top_k_sensitivity,
    summarize,
)


def run_self_checks() -> None:
    cfg = JointConfig(trace_length=100, trace_start=94, offset_step=4, network_repeats=1, tasks_per_network=10)
    actual, predictions = load_inputs(Path(__file__).resolve().parents[1] / "test")
    nodes_actual, nodes_predictions = build_node_replays(actual, predictions, cfg)
    assert nodes_actual.shape == (10, 100, 2)
    assert nodes_predictions.shape == (10, 100, 6, 2)
    replays = build_cache_replays(nodes_actual, nodes_predictions, cfg)
    assert set(replays) == {"proposed", "last_mpc", "fits_periodic", "fits_delta", "fits_gossip", "oracle_state"}
    assert np.isclose(replays["proposed"].update_rate, 0.5)
    assert np.isclose(replays["last_mpc"].update_rate, 0.5)
    assert np.isclose(replays["fits_periodic"].update_rate, 0.5)
    assert np.isclose(replays["fits_delta"].update_rate, 0.5)
    assert replays["oracle_state"].cache_wrmse == 0.0
    tasks = generate_tasks(nodes_actual, cfg, seed=1)
    assert len(tasks) == 10
    for time_slot, origin, demand in tasks:
        availability = 100.0 - nodes_actual[:, time_slot]
        assert any(node != origin and np.all(availability[node] >= demand) for node in range(cfg.num_nodes))
    graph, _, _ = graph_and_trust(cfg, 2)
    remote_tasks = generate_tasks(nodes_actual, cfg, seed=3, graph=graph, remote_required=True)
    for time_slot, origin, demand in remote_tasks:
        availability = 100.0 - nodes_actual[:, time_slot]
        assert not any(np.all(availability[node] >= demand) for node in graph[origin])
        assert any(
            node != origin and node not in graph[origin] and np.all(availability[node] >= demand)
            for node in range(cfg.num_nodes)
        )
    raw = run_joint_trials(nodes_actual, replays, cfg, pipelines=["proposed", "fits_periodic"])
    assert len(raw) == 2 * cfg.tasks_per_network * 2
    assert (raw.true_positive_candidates + raw.false_negative_candidates == raw.true_candidate_count).all()
    assert (raw.true_positive_candidates + raw.false_positive_candidates == raw.cached_candidate_count).all()
    summary = summarize(raw)
    assert np.allclose(
        summary.candidate_micro_recall + summary.candidate_false_negative_rate,
        1.0,
    )
    top_raw, top_summary, top_tests = run_top_k_sensitivity(
        nodes_actual, replays, cfg, top_k_values=(1, 3)
    )
    assert len(top_raw) == 2 * cfg.tasks_per_network * 2
    assert set(top_summary.top_k) == {1, 3}
    assert len(top_tests) == 2
    print("20 joint-experiment self-checks passed")


if __name__ == "__main__":
    run_self_checks()
