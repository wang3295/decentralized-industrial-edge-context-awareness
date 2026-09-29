"""Run a fixed-trace conformance experiment for simulation/deployment updates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import sys
from typing import Any

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


def fixed_trace(length: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    slots = np.arange(length, dtype=float)
    actual = np.column_stack(
        [
            42.0 + 19.0 * np.sin(slots / 3.7) + 5.0 * np.sin(slots / 1.3),
            51.0 + 14.0 * np.cos(slots / 4.1) + 4.0 * np.sin(slots / 1.9),
        ]
    )
    actual = np.clip(actual, 0.0, 100.0)
    forecasts = np.empty((length, horizon, 2), dtype=float)
    for t in range(length):
        for h in range(horizon):
            source = min(t + h, length - 1)
            bias = np.array([0.35 * h, -0.2 * h])
            forecasts[t, h] = np.clip(actual[source] + bias, 0.0, 100.0)
    return actual, forecasts


def strategy_conformance(
    node_module: Any,
    strategy: str,
    actual: np.ndarray,
    forecasts: np.ndarray,
    budget: float,
    horizon: int,
) -> dict[str, Any]:
    agent = node_module.NodeAgent.__new__(node_module.NodeAgent)
    agent.update_strategy = strategy
    agent.budget = budget
    agent.delta_threshold = 5.0
    agent.eta = 1.0
    agent.weights = np.array([0.5, 0.5])
    agent.last_sent = None
    agent.budget_guard = node_module.TokenBudget(budget, epoch_slots=horizon)

    shared_guard = RepeatedEpochBudgetGuard(budget, epoch_slots=horizon)
    shared_last_sent = None
    proposal_mismatches = 0
    action_mismatches = 0
    node_actions: list[int] = []
    shared_actions: list[int] = []

    for t, current in enumerate(actual):
        node_proposal = agent._propose_update(current, forecasts[t])
        shared_proposal = runtime_update_proposal(
            strategy=strategy,
            current=current,
            forecast=forecasts[t],
            last_sent=shared_last_sent,
            budget=budget,
            epoch_position=shared_guard.epoch_position,
            delta_threshold=5.0,
            eta=1.0,
            weights=np.array([0.5, 0.5]),
        )
        proposal_mismatches += int(node_proposal != shared_proposal)
        node_action = agent.budget_guard.allow(node_proposal)
        shared_action = shared_guard.allow(shared_proposal)
        action_mismatches += int(node_action != shared_action)
        node_actions.append(int(node_action))
        shared_actions.append(int(shared_action))
        if node_action:
            agent.last_sent = current.copy()
        if shared_action:
            shared_last_sent = current.copy()

    return {
        "strategy": strategy,
        "slots": len(actual),
        "proposal_mismatches": proposal_mismatches,
        "action_mismatches": action_mismatches,
        "node_update_count": sum(node_actions),
        "shared_update_count": sum(shared_actions),
        "action_sequence_identical": node_actions == shared_actions,
        "action_sequence": "".join(map(str, node_actions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "shared_core_conformance_20260908",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    node_module = load_module(
        "shared_conformance_node_agent",
        RAFT_ROOT / "testbed_deployment" / "node_agent.py",
    )
    simulation_module = load_module(
        "shared_conformance_simulation",
        Path(__file__).with_name("run_update_experiments.py"),
    )
    core_path = RAFT_ROOT / "edgebench_core" / "update_policies.py"
    core_hash = hashlib.sha256(core_path.read_bytes()).hexdigest()

    budget = 0.5
    horizon = 6
    actual, forecasts = fixed_trace(length=60, horizon=horizon)
    rows = [
        strategy_conformance(node_module, strategy, actual, forecasts, budget, horizon)
        for strategy in ("mpc_dp", "periodic", "send_on_delta")
    ]

    all_passed = all(
        row["proposal_mismatches"] == 0
        and row["action_mismatches"] == 0
        and row["node_update_count"] == 30
        and row["shared_update_count"] == 30
        for row in rows
    )
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "slots": len(actual),
        "horizon": horizon,
        "budget": budget,
        "expected_updates": 30,
        "shared_core_sha256": core_hash,
        "node_imports_shared_mpc": node_module.mpc_dp_first_action is mpc_dp_first_action,
        "simulation_imports_shared_mpc": simulation_module.mpc_dp_first_action is mpc_dp_first_action,
        "node_guard": node_module.TokenBudget.__name__,
        "simulation_guard": simulation_module.BudgetGuard.__name__,
        "all_checks_passed": all_passed,
    }

    with (args.output_dir / "conformance_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "conformance_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table_rows = "\n".join(
        f"| {row['strategy']} | {row['proposal_mismatches']} | {row['action_mismatches']} | "
        f"{row['node_update_count']} | {row['shared_update_count']} |"
        for row in rows
    )
    report = f"""# 仿真—实物更新策略共享核心一致性验证

## 1. 验证目的

本验证不是重新比较算法优劣，而是检查仿真程序和实物节点代理是否调用同一个MPC-DP决策实现，并确认部署适配器在固定输入下不会改变逐时隙更新决策。

## 2. 固定协议

- 资源维度：CPU、内存；权重均为0.5。
- 固定轨迹：60个时隙；预测窗口6。
- 更新预算：B=0.5，即每6时隙准确发送3次，共30次。
- 策略：MPC-DP、Periodic、Send-on-delta。
- MPC-DP参数：eta=1.0；Send-on-delta阈值=5.0。
- 比较对象：`NodeAgent._propose_update`与共享运行时函数逐时隙输出。

## 3. 结果

| 策略 | proposal不一致数 | 最终action不一致数 | 节点代理更新数 | 共享核心更新数 |
|---|---:|---:|---:|---:|
{table_rows}

全部检查通过：**{'是' if all_passed else '否'}**。

- 节点代理与仿真程序引用同一个MPC-DP函数对象：**{'是' if metadata['node_imports_shared_mpc'] and metadata['simulation_imports_shared_mpc'] else '否'}**。
- 共享核心SHA-256：`{core_hash}`。
- 实物在线保护器：`{metadata['node_guard']}`。
- 历史统一轨迹实验保护器：`{metadata['simulation_guard']}`。

## 4. 结论边界

验证说明MPC-DP目标函数已有单一代码来源，实物节点的策略适配没有产生实现漂移。两类实验仍可采用不同预算时间尺度：历史仿真在整段轨迹上执行精确预算，实物在线系统每6时隙重复执行预算。两者总预算可相同，但发送时刻不必完全相同；论文必须将其写成协议差异，不能宣称两组结果来自完全相同的时间尺度。

本验证也不重新证明MPC-DP优于Periodic或Send-on-delta；算法优劣仍以独立的效果实验与统计检验为准。
"""
    (args.output_dir / "仿真实物共享核心一致性验证.md").write_text(report, encoding="utf-8")

    print(json.dumps({"metadata": metadata, "results": rows}, ensure_ascii=False, indent=2))
    if not all_passed:
        raise SystemExit("conformance checks failed")


if __name__ == "__main__":
    main()
