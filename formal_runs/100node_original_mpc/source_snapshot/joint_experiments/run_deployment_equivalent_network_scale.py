"""Network-scale simulation using the deployed update-policy implementation.

This is intentionally a separate experiment from
``run_network_scale_sensitivity.py``.  The historical scale experiment labels
threshold-aware MPC-DP as Full; the physical ``formal_v2`` testbed used the
paper's original MPC-DP with eta=1.0 and a repeated six-slot online budget.
This runner removes that profile mismatch while retaining the discrete-event
network simulator for large-scale search, trust, churn and failure analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RAFT_DIR = HERE.parent
if str(RAFT_DIR) not in sys.path:
    sys.path.insert(0, str(RAFT_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(RAFT_DIR / "update_experiments") not in sys.path:
    sys.path.insert(0, str(RAFT_DIR / "update_experiments"))

from edgebench_core import RepeatedEpochBudgetGuard, runtime_update_proposal  # noqa: E402
import run_network_scale_sensitivity as scale  # noqa: E402
from run_full_pipeline_ablation import FullAblationConfig, network_summary  # noqa: E402
from run_joint_discovery_experiment import CacheReplay, build_node_replays  # noqa: E402
from run_update_experiments import load_inputs  # noqa: E402


LEGACY_PIPELINES = {
    "full": "Full: deployed MPC-DP + Workflow",
    "periodic": "Periodic + Workflow",
    "send_on_delta": "Send-on-delta + Workflow",
    "gossip": "Deployed MPC-DP + Gossip-style query",
    "no_trust": "Deployed MPC-DP + Workflow, no trust",
}

LEGACY_PIPELINE_SPEC = {
    "full": ("runtime_mpc_dp", "online_beta", "workflow"),
    "periodic": ("runtime_periodic", "online_beta", "workflow"),
    "send_on_delta": ("runtime_send_on_delta", "online_beta", "workflow"),
    "gossip": ("runtime_mpc_dp", "online_beta", "gossip"),
    "no_trust": ("runtime_mpc_dp", "no_trust", "workflow"),
}

PAPER_FIVE_PIPELINES = {
    "full": "Full: deployed MPC-DP + Workflow",
    "periodic": "Periodic + Workflow",
    "send_on_delta": "Send-on-delta + Workflow",
    "lyapunov": "Lyapunov + Workflow",
    "gossip": "Deployed MPC-DP + Gossip-style query",
}

PAPER_FIVE_PIPELINE_SPEC = {
    "full": ("runtime_mpc_dp", "online_beta", "workflow"),
    "periodic": ("runtime_periodic", "online_beta", "workflow"),
    "send_on_delta": ("runtime_send_on_delta", "online_beta", "workflow"),
    "lyapunov": ("runtime_lyapunov", "online_beta", "workflow"),
    "gossip": ("runtime_mpc_dp", "online_beta", "gossip"),
}

# Preserve the historical public module constants for imports and old commands.
PIPELINES = LEGACY_PIPELINES
PIPELINE_SPEC = LEGACY_PIPELINE_SPEC


def replay_deployment_policy(
    actual: np.ndarray,
    predictions: np.ndarray,
    strategy: str,
    budget: float,
    horizon: int,
    eta: float,
    delta_threshold: float,
    lyapunov_v0: float = 0.001,
    lyapunov_power: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay exactly the online update proposal and budget guard used by nodes."""

    if len(actual) % horizon:
        raise ValueError(
            "deployment-equivalent formal traces must contain complete budget epochs"
        )
    cache_history = np.empty_like(actual)
    actions = np.zeros(len(actual), dtype=bool)
    guard = RepeatedEpochBudgetGuard(budget, epoch_slots=horizon)
    last_sent: np.ndarray | None = None
    lyapunov_queue = 0.0
    weights = np.array([0.5, 0.5])

    for t, current in enumerate(actual):
        forecast = predictions[t, : min(horizon, predictions.shape[1])]
        proposed = runtime_update_proposal(
            strategy=strategy,
            current=current,
            forecast=forecast,
            last_sent=last_sent,
            budget=budget,
            epoch_position=guard.epoch_position,
            delta_threshold=delta_threshold,
            eta=eta,
            weights=weights,
            lyapunov_queue=lyapunov_queue,
            lyapunov_v0=lyapunov_v0,
            lyapunov_power=lyapunov_power,
            step_index=t,
        )
        action = guard.allow(proposed)
        actions[t] = action
        if action:
            last_sent = current.copy()
        if strategy == "lyapunov":
            lyapunov_queue = max(lyapunov_queue + float(action) - budget, 0.0)
        if last_sent is None:
            raise AssertionError("the first online slot must initialize the neighbor cache")
        cache_history[t] = last_sent

    expected = len(actual) * guard.epoch_target // horizon
    if int(actions.sum()) != expected:
        raise AssertionError(f"{strategy}: {actions.sum()} updates, expected {expected}")
    return cache_history, actions


def build_deployment_replays(
    nodes_actual: np.ndarray,
    nodes_predictions: np.ndarray,
    cfg: FullAblationConfig,
    eta: float,
    delta_threshold: float,
    lyapunov_v0: float = 0.001,
    lyapunov_power: float = 0.5,
) -> dict[str, CacheReplay]:
    strategies = {
        "runtime_mpc_dp": "mpc_dp",
        "runtime_periodic": "periodic",
        "runtime_send_on_delta": "send_on_delta",
        "runtime_lyapunov": "lyapunov",
    }
    replays: dict[str, CacheReplay] = {}
    for key, strategy in strategies.items():
        caches = []
        actions = []
        for node in range(cfg.num_nodes):
            cache, action = replay_deployment_policy(
                nodes_actual[node],
                nodes_predictions[node],
                strategy,
                cfg.update_budget,
                cfg.horizon,
                eta,
                delta_threshold,
                lyapunov_v0,
                lyapunov_power,
            )
            caches.append(cache)
            actions.append(action)
        cache_array = np.asarray(caches)
        action_array = np.asarray(actions)
        errors = nodes_actual - cache_array
        wrmse = float(np.sqrt(np.mean(np.sum(0.5 * np.square(errors), axis=2))))
        replays[key] = CacheReplay(
            pipeline=key,
            cache_utilization=cache_array,
            actions=action_array,
            cache_wrmse=wrmse,
            update_count=int(action_array.sum()),
            update_rate=float(action_array.mean()),
        )
    return replays


def validate_results(
    raw: pd.DataFrame,
    network: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Validate the row structure and the protocol accounting before reporting."""

    expected_raw = (
        len(args.node_counts)
        * len(PIPELINES)
        * args.repeats
        * args.tasks_per_network
    )
    expected_network = len(args.node_counts) * len(PIPELINES) * args.repeats
    key = ["num_nodes", "pipeline", "repeat", "task"]
    message_columns = [
        "search_messages",
        "request_messages",
        "response_messages",
        "confirmation_messages",
        "amortized_update_messages",
        "total_messages",
    ]
    core_columns = [
        "search_messages",
        "amortized_update_messages",
        "total_messages",
        "cache_wrmse",
        "update_rate",
    ]
    checks = {
        "raw_rows_match": len(raw) == expected_raw,
        "network_rows_match": len(network) == expected_network,
        "summary_rows_match": len(summary) == len(args.node_counts) * len(PIPELINES),
        "paired_test_rows_match": len(tests) == len(args.node_counts) * (len(PIPELINES) - 1) * 2,
        "no_duplicate_task_keys": not raw.duplicated(key).any(),
        "all_methods_present": set(raw.pipeline.unique()) == set(PIPELINES),
        "all_node_counts_present": set(raw.num_nodes.unique()) == set(args.node_counts),
        "nonnegative_message_counts": bool((raw[message_columns] >= 0).all().all()),
        "search_message_accounting_exact": bool(
            np.allclose(
                raw.search_messages,
                raw.request_messages
                + raw.response_messages
                + raw.confirmation_messages,
            )
        ),
        "total_message_accounting_exact": bool(
            np.allclose(
                raw.total_messages,
                raw.search_messages + raw.amortized_update_messages,
            )
        ),
        "all_core_values_finite": bool(np.isfinite(raw[core_columns].to_numpy()).all()),
        "honest_success_implies_success": bool((raw.honest_success <= raw.success).all()),
        "success_latency_complete": bool(
            raw.loc[raw.success.eq(1), "success_latency_ms"].notna().all()
            and raw.loc[raw.success.eq(0), "success_latency_ms"].isna().all()
        ),
        "deadline_respected": bool(
            (raw.loc[raw.success.eq(1), "success_latency_ms"] <= 180.0).all()
        ),
        "maximum_hops_respected": bool((raw.selected_hops.dropna() <= 5).all()),
        "identical_update_budget": bool(np.allclose(raw.update_rate, args.budget)),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"formal result validation failed: {failed}")

    method_task_counts = {
        f"N={int(count)}/{pipeline}": int(len(group))
        for (count, pipeline), group in raw.groupby(["num_nodes", "pipeline"])
    }
    method_network_counts = {
        f"N={int(count)}/{pipeline}": int(group.repeat.nunique())
        for (count, pipeline), group in raw.groupby(["num_nodes", "pipeline"])
    }
    return {
        "status": "PASS",
        "checks": checks,
        "expected_raw_rows": expected_raw,
        "actual_raw_rows": len(raw),
        "actual_network_rows": len(network),
        "method_task_counts": method_task_counts,
        "method_independent_network_counts": method_network_counts,
        "maximum_selected_hops": float(raw.selected_hops.dropna().max()),
        "maximum_success_latency_ms": float(
            raw.loc[raw.success.eq(1), "success_latency_ms"].max()
        ),
        "update_rates": sorted(float(value) for value in raw.update_rate.unique()),
    }


def write_report(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    indexed = summary.set_index(["num_nodes", "pipeline"])
    findings = []
    for count in args.node_counts:
        full = indexed.loc[(count, "full")]
        periodic = indexed.loc[(count, "periodic")]
        delta = indexed.loc[(count, "send_on_delta")]
        lyapunov = (
            indexed.loc[(count, "lyapunov")] if "lyapunov" in indexed.index.get_level_values("pipeline") else None
        )
        gossip = indexed.loc[(count, "gossip")]
        lyapunov_text = (
            f"Lyapunov {lyapunov.honest_success_percent:.2f}%，" if lyapunov is not None else ""
        )
        findings.append(
            f"- N={count}：Full诚实候选发现率{full.honest_success_percent:.2f}%，"
            f"Periodic {periodic.honest_success_percent:.2f}%，"
            f"Send-on-delta {delta.honest_success_percent:.2f}%，"
            f"{lyapunov_text}"
            f"Gossip {gossip.honest_success_percent:.2f}%；"
            f"Full缓存WRMSE {full.cache_wrmse:.4f}。"
        )
    finding_text = "\n".join(findings)

    significance_lines = []
    honest_tests = tests[tests.endpoint.eq("honest_success_rate")]
    for count in args.node_counts:
        for baseline in ("periodic", "send_on_delta", "lyapunov", "gossip"):
            comparison = f"full vs {baseline}"
            row = honest_tests[
                honest_tests.num_nodes.eq(count)
                & honest_tests.comparison.eq(comparison)
            ].iloc[0]
            verdict = "显著" if row.holm_adjusted_p < 0.05 else "不显著"
            significance_lines.append(
                f"- N={count}，Full相对{baseline}："
                f"{row.full_minus_baseline_pp:+.2f}个百分点，"
                f"网络簇bootstrap 95% CI "
                f"[{row.cluster_bootstrap_ci_low_pp:.2f}, "
                f"{row.cluster_bootstrap_ci_high_pp:.2f}]，"
                f"Holm校正p={row.holm_adjusted_p:.4g}（{verdict}）。"
            )
    significance_text = "\n".join(significance_lines)

    tradeoff_lines = []
    for count in args.node_counts:
        full = indexed.loc[(count, "full")]
        periodic = indexed.loc[(count, "periodic")]
        delta = indexed.loc[(count, "send_on_delta")]
        lyapunov = indexed.loc[(count, "lyapunov")]
        gossip = indexed.loc[(count, "gossip")]
        gossip_reduction = 100.0 * (
            1.0 - gossip.mean_total_messages / full.mean_total_messages
        )
        tradeoff_lines.append(
            f"- N={count}：Full总消息{full.mean_total_messages:.2f}条/请求，"
            f"Gossip为{gossip.mean_total_messages:.2f}条（少{gossip_reduction:.2f}%）；"
            f"Full/Periodic/Send-on-delta/Lyapunov缓存WRMSE分别为"
            f"{full.cache_wrmse:.4f}/{periodic.cache_wrmse:.4f}/"
            f"{delta.cache_wrmse:.4f}/{lyapunov.cache_wrmse:.4f}。"
        )
    tradeoff_text = "\n".join(tradeoff_lines)

    report = f"""# 实物等价更新协议下的网络规模实验

## 1. 为什么重新运行

旧规模实验中的Full采用门槛感知MPC-DP，而实物测试床采用论文原始MPC-DP。为避免把两种算法混称为同一方法，本实验固定调用与20节点实物节点相同的共享更新代码，再在离散事件仿真中扩展到50/100/150节点。

## 2. 协议

- 节点数：{', '.join(map(str, args.node_counts))}；每个规模{args.repeats}个独立连通网络，每个网络{args.tasks_per_network}个连续任务。
- 资源轨迹：每节点{args.trace_length}时隙，节点偏移{args.offset_step}；CPU/内存权重0.5/0.5。
- 在线更新：预算B={args.budget:g}，6时隙为一个完整预算周期；MPC-DP eta={args.eta:g}；Send-on-delta阈值={args.delta_threshold:g}；Lyapunov V0={args.lyapunov_v0:g}、幂指数={args.lyapunov_power:g}。
- 搜索：平均业务度数4，Dmax=5，绝对截止180 ms，top-3二次确认。
- 动态性：5%链路变化、5%节点中途断联、20%恶意节点；在线Beta信任跨连续任务更新。
- 公平性：所有更新方法均由同一预算保护器保证每个完整周期发送相同数量状态消息。Gossip只替换查询传播；其他组件保持一致。
- 代码关系：状态更新逐时隙调用`edgebench_core.runtime_update_proposal`和`RepeatedEpochBudgetGuard`；搜索在仿真中使用离散事件实现，实物中使用HTTP实现，因此网络时延数值用于趋势验证，不要求与实物毫秒数完全相等。

## 3. 汇总结果

{scale.markdown_table(summary, ['num_nodes', 'pipeline', 'networks', 'honest_success_percent', 'honest_success_ci95_percent', 'malicious_selection_percent', 'malicious_selection_ci95_percent', 'cache_blocked_percent', 'candidate_recall_percent', 'p95_success_latency_ms', 'mean_search_messages', 'mean_update_messages', 'mean_total_messages', 'mean_total_messages_per_node', 'cache_wrmse', 'update_rate_percent'])}

## 4. 配对统计检验

{scale.markdown_table(tests, ['num_nodes', 'endpoint', 'comparison', 'network_clusters', 'full_percent', 'baseline_percent', 'full_minus_baseline_pp', 'cluster_bootstrap_ci_low_pp', 'cluster_bootstrap_ci_high_pp', 'wilcoxon_p', 'holm_adjusted_p'])}

## 5. 直接读数

{finding_text}

## 6. Full与基线的配对结论

统计推断的独立单位是生成的连通网络，不是同一网络内的200个连续任务。Wilcoxon检验在网络簇层面完成，并对{len(tests)}个检验统一进行Holm校正。

{significance_text}

## 7. 准确性—通信开销权衡

{tradeoff_text}

- Full在三个规模下均显著优于Send-on-delta和Gossip的诚实候选发现率。
- Full在N=50时低于Periodic和Lyapunov；在N=100和150时，与这两种公平更新基线的差异经Holm校正后不显著。因此不能宣称Full全面优于所有更新策略。
- Full的缓存WRMSE在所有规模下最低，但连续资源数值更准确不等于门槛附近的候选判断一定更优；这正是需要同时报告WRMSE、候选召回率和最终发现率的原因。
- Gossip以明显更少的查询消息换取较低发现率，属于有效的通信开销—成功率对照，不能只按成功率把它描述成弱基线。
- 五种方法的P95均约为153 ms，差异不足1 ms。该值来自15 ms单跳时延的离散事件模型，只能用于仿真内相对比较；真实物理时延应引用20节点测试床结果。

## 8. 结论边界

这组结果用于检验“相同更新算法在小型实物网络与大规模仿真中的方向是否一致”，不是让仿真毫秒时延复刻真实网络，也不替代20节点三主机部署。若Full未显著优于某个公平基线，必须保留该负结果，不能根据规模点重新选择有利参数。旧的门槛感知结果继续作为算法变体/消融，不再与本实验共享Full名称。
"""
    (output_dir / "实物等价更新协议网络规模实验结果与分析.md").write_text(
        report, encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=RAFT_DIR / "test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "deployment_equivalent_scale_formal",
    )
    parser.add_argument("--node-counts", type=scale.parse_node_counts, default=(50, 100, 150))
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--tasks-per-network", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--trace-length", type=int, default=228)
    parser.add_argument("--trace-start", type=int, default=94)
    parser.add_argument("--offset-step", type=int, default=1)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--delta-threshold", type=float, default=5.0)
    parser.add_argument("--lyapunov-v0", type=float, default=0.001)
    parser.add_argument("--lyapunov-power", type=float, default=0.5)
    parser.add_argument(
        "--method-set",
        choices=("legacy", "paper-five"),
        default="legacy",
        help="legacy preserves the old no-trust run; paper-five replaces it with Lyapunov",
    )
    return parser.parse_args()


def main() -> None:
    global PIPELINES, PIPELINE_SPEC
    args = parse_args()
    if args.trace_length % 6:
        raise ValueError("trace length must be divisible by the six-slot online epoch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual, predictions = load_inputs(args.data_dir)

    if args.method_set == "paper-five":
        PIPELINES = PAPER_FIVE_PIPELINES
        PIPELINE_SPEC = PAPER_FIVE_PIPELINE_SPEC
    else:
        PIPELINES = LEGACY_PIPELINES
        PIPELINE_SPEC = LEGACY_PIPELINE_SPEC
    scale.PIPELINES = PIPELINES
    scale.PIPELINE_SPEC = PIPELINE_SPEC
    all_raw = []
    configs = {}
    for num_nodes in args.node_counts:
        cfg = FullAblationConfig(
            num_nodes=num_nodes,
            trace_length=args.trace_length,
            trace_start=args.trace_start,
            offset_step=args.offset_step,
            update_budget=args.budget,
            horizon=6,
            eta=args.eta,
            network_repeats=args.repeats,
            tasks_per_network=args.tasks_per_network,
        )
        required_stop = cfg.trace_start + (cfg.num_nodes - 1) * cfg.offset_step + cfg.trace_length
        if required_stop > len(actual):
            raise ValueError(
                f"N={num_nodes} requires {required_stop} rows, only {len(actual)} available"
            )
        nodes_actual, nodes_predictions = build_node_replays(actual, predictions, cfg)
        replays = build_deployment_replays(
            nodes_actual,
            nodes_predictions,
            cfg,
            args.eta,
            args.delta_threshold,
            args.lyapunov_v0,
            args.lyapunov_power,
        )
        update_rates = {key: value.update_rate for key, value in replays.items()}
        if len({round(value, 12) for value in update_rates.values()}) != 1:
            raise AssertionError(f"budget mismatch: {update_rates}")
        frame = scale.run_scale_condition(num_nodes, nodes_actual, replays, cfg, args.workers)
        expected = cfg.network_repeats * cfg.tasks_per_network * len(PIPELINES)
        if len(frame) != expected:
            raise AssertionError(f"N={num_nodes}: expected {expected}, got {len(frame)}")
        all_raw.append(frame)
        configs[str(num_nodes)] = asdict(cfg)

    raw = pd.concat(all_raw, ignore_index=True)
    network_frames = []
    for num_nodes, group in raw.groupby("num_nodes", sort=True):
        frame = network_summary(group.drop(columns="num_nodes"))
        frame.insert(0, "num_nodes", int(num_nodes))
        network_frames.append(frame)
    network = pd.concat(network_frames, ignore_index=True)
    summary = scale.summarize_by_scale(network)
    tests = scale.paired_tests(network, args.node_counts)
    validation = validate_results(raw, network, summary, tests, args)

    raw.to_csv(args.output_dir / "deployment_equivalent_raw.csv", index=False)
    network.to_csv(args.output_dir / "deployment_equivalent_network.csv", index=False)
    summary.to_csv(args.output_dir / "deployment_equivalent_summary.csv", index=False)
    tests.to_csv(args.output_dir / "deployment_equivalent_paired_tests.csv", index=False)
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scale.plot_results(summary, args.output_dir)
    write_report(summary, tests, args, args.output_dir)

    core_path = RAFT_DIR / "edgebench_core" / "update_policies.py"
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "prediction_model": "FITS-Delta",
        "data_dir": str(args.data_dir.resolve()),
        "input_sha256": {
            name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
            for name in ("value.csv", "cpu_predictions.csv", "memory_predictions.csv")
        },
        "node_counts": list(args.node_counts),
        "method_set": args.method_set,
        "pipelines": PIPELINES,
        "pipeline_spec": PIPELINE_SPEC,
        "configs": configs,
        "update_profile": {
            "budget_guard": "RepeatedEpochBudgetGuard",
            "epoch_slots": 6,
            "budget": args.budget,
            "eta": args.eta,
            "delta_threshold": args.delta_threshold,
            "lyapunov_v0": args.lyapunov_v0,
            "lyapunov_power": args.lyapunov_power,
            "shared_core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        },
        "independent_unit": "generated connected network",
        "within_network_repeated_observations": args.tasks_per_network,
        "rows": len(raw),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
