import numpy as np

from run_trust_robustness_experiment import (
    TrustExperimentConfig,
    holm_adjust,
    initial_beta_state,
    joint_config,
    policy_score,
    run_trust_ranked_search,
    trust_state_metrics,
)
from run_dynamic_search_experiments import (
    SearchConfig,
    create_connected_business_graph,
    generate_environment,
)


def main() -> None:
    cfg = TrustExperimentConfig(network_repeats=2, tasks_per_network=8)
    jcfg = joint_config(cfg)
    assert jcfg.num_nodes == 20
    assert jcfg.offset_step == 4
    assert jcfg.update_budget == 0.5

    graph = create_connected_business_graph(20, 4, np.random.default_rng(4))
    malicious = np.zeros(20, dtype=bool)
    malicious[:4] = True
    reliability = np.where(malicious, 0.2, 0.95)
    alpha, beta = initial_beta_state(graph, reliability, 8, history_length=0)
    assert set(alpha) == set(beta)
    assert all(alpha[key] >= 1.0 and beta[key] >= 1.0 for key in alpha)
    assert all(alpha[key] == 1.0 and beta[key] == 1.0 for key in alpha)
    honest, bad, mae, auc = trust_state_metrics(
        graph, alpha, beta, malicious, reliability
    )
    assert 0.0 <= honest <= 1.0
    assert 0.0 <= bad <= 1.0
    assert 0.0 <= mae <= 1.0
    assert 0.0 <= auc <= 1.0

    adjusted = holm_adjust([0.001, 0.01, 0.2])
    assert np.all(adjusted >= np.array([0.001, 0.01, 0.2]))
    assert np.all(adjusted <= 1.0)

    small_cfg = SearchConfig(
        num_nodes=20,
        average_degree=4,
        max_hops=5,
        deadline_ms=180.0,
        top_k=3,
    )
    env = generate_environment(1234, small_cfg)
    reliability_small = np.full(small_cfg.num_nodes, 0.95)
    malicious_small = np.zeros(small_cfg.num_nodes, dtype=bool)
    valid_small = np.ones(small_cfg.num_nodes, dtype=bool)
    for index, dissemination in enumerate(
        ("workflow", "gossip", "random_walk", "expanding_ring", "centralized_oracle")
    ):
        result = run_trust_ranked_search(
            env,
            small_cfg,
            "oracle" if dissemination == "centralized_oracle" else "online_beta",
            reliability_small,
            malicious_small,
            valid_small,
            seed=5000 + index,
            collection_ms=120.0,
            dissemination=dissemination,
        )
        assert result.message_count == (
            result.request_messages
            + result.response_messages
            + result.confirmation_messages
        )
        assert result.duplicate_requests >= 0
        assert np.isnan(result.selected_hops) or result.selected_hops <= small_cfg.max_hops

    candidate = next(iter(env.known_adj[env.origin]))
    direct_score = policy_score(
        "direct_only",
        env,
        candidate,
        [env.origin, candidate],
        reliability_small,
        small_cfg,
    )
    assert 0.0 < direct_score < 1.0
    print("33 trust-robustness self-checks passed")


if __name__ == "__main__":
    main()
