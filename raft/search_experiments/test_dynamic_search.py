import numpy as np

from run_dynamic_search_experiments import (
    SearchConfig,
    generate_environment,
    path_trust,
    run_search,
)


def run_self_checks() -> None:
    cfg = SearchConfig(num_nodes=30, average_degree=4, outage_rate=0.0, topology_churn_rate=0.0)
    env = generate_environment(42, cfg)
    assert all(len(nbrs) > 0 for nbrs in env.known_adj)
    assert all(u in env.known_adj[v] for u, nbrs in enumerate(env.known_adj) for v in nbrs)
    assert not any(np.all(env.cache_resource[v] >= env.demand) for v in env.known_adj[env.origin])
    assert all(0.0 < path_trust(env, [u, v], cfg.gamma) < 1.0 for u in range(cfg.num_nodes) for v in env.known_adj[u])

    workflow = run_search("workflow_confirm", env, cfg, seed=100)
    gossip = run_search("gossip_confirm", env, cfg, seed=101)
    no_confirm = run_search("workflow_no_confirm", env, cfg, seed=102)
    for result in [workflow, gossip, no_confirm]:
        assert result.message_count >= 0
        assert result.decision_latency_ms <= cfg.deadline_ms
        assert result.success in (0, 1)
    assert workflow.confirmation_messages == 2 * workflow.confirmation_attempts

    churn_cfg = SearchConfig(num_nodes=30, average_degree=4, topology_churn_rate=0.2)
    churn_env = generate_environment(43, churn_cfg)
    known_edges = {(min(u, v), max(u, v)) for u, nbrs in enumerate(churn_env.known_adj) for v in nbrs}
    actual_edges = {(min(u, v), max(u, v)) for u, nbrs in enumerate(churn_env.actual_adj) for v in nbrs}
    assert known_edges != actual_edges
    print("7 dynamic-search self-checks passed")


if __name__ == "__main__":
    run_self_checks()
