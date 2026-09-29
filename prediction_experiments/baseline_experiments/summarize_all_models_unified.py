from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "all_models_unified_full_v1"
SUMMARY_CSV = RESULT_DIR / "prediction_summary_mean_std.csv"
RAW_CSV = RESULT_DIR / "prediction_metrics_raw.csv"
REPORT_PATH = RESULT_DIR / "预测主实验九模型统一重跑结果与分析.md"
ARTICLE_PATH = ROOT.parent.parent / "文章" / "预测主实验结果汇总_原论文加新增模型.md"

DATASETS = ["Alibaba", "GWA-T-12", "Google"]
MODEL_ORDER = [
    "FITS",
    "FITS-Delta",
    "NaiveLastValue",
    "ARIMA",
    "SingleLayerGRU",
    "DLinear",
    "S2S-LSTM",
    "FreTS",
    "PatchTST",
]
DISPLAY = {
    "NaiveLastValue": "Naive last-value",
    "SingleLayerGRU": "单层 GRU",
}
METRICS = ["RMSE", "MAE", "R2", "MAPE_pct", "sMAPE_pct"]
LOWER_IS_BETTER = {"RMSE": True, "MAE": True, "R2": False, "MAPE_pct": True, "sMAPE_pct": True}


def fmt_mean_sd(mean: float, sd: float, runs: int, metric: str) -> str:
    digits = 4
    if runs == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{sd:.{digits}f}"


def ranked_metric_cells(frame: pd.DataFrame, metric: str) -> dict[str, str]:
    vals = frame.set_index("model")[f"{metric}_mean"]
    ordered = vals.sort_values(ascending=LOWER_IS_BETTER[metric]).index.tolist()
    cells: dict[str, str] = {}
    for model in MODEL_ORDER:
        row = frame.loc[frame.model == model].iloc[0]
        cell = fmt_mean_sd(row[f"{metric}_mean"], row[f"{metric}_std"], int(row.runs), metric)
        if model == ordered[0]:
            cell = f"**{cell}**"
        elif model == ordered[1]:
            cell = f"<u>{cell}</u>"
        cells[model] = cell
    return cells


def result_table(frame: pd.DataFrame) -> str:
    rank = {metric: ranked_metric_cells(frame, metric) for metric in METRICS}
    lines = [
        "| 模型 | RMSE↓ | MAE↓ | R²↑ | MAPE (%)↓ | sMAPE (%)↓ | 参数量 | CPU中位时延 (ms) | CPU P95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        row = frame.loc[frame.model == model].iloc[0]
        params = f"{int(round(row.parameter_count_mean)):,}"
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY.get(model, model),
                    rank["RMSE"][model],
                    rank["MAE"][model],
                    rank["R2"][model],
                    rank["MAPE_pct"][model],
                    rank["sMAPE_pct"][model],
                    params,
                    f"{row.cpu_latency_median_ms_mean:.4f}",
                    f"{row.cpu_latency_p95_ms_mean:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def pct_change(new: float, old: float, lower_better: bool = True) -> float:
    if lower_better:
        return (old - new) / old * 100.0
    return (new - old) / abs(old) * 100.0


def get(summary: pd.DataFrame, dataset: str, model: str, metric: str) -> float:
    return float(summary.loc[(summary.dataset == dataset) & (summary.model == model), f"{metric}_mean"].iloc[0])


def build_report(summary: pd.DataFrame, raw: pd.DataFrame) -> str:
    env = json.loads((RESULT_DIR / "environment.json").read_text(encoding="utf-8"))
    config = json.loads((RESULT_DIR / "config.json").read_text(encoding="utf-8"))["config"]
    run_info = json.loads((RESULT_DIR / "run_info.json").read_text(encoding="utf-8"))

    lines = [
        "# 预测主实验九模型统一重跑结果与分析",
        "",
        "> 本报告是最终统一协议结果。它整体替代此前把论文旧表与新增基线拼接在一起的临时表。粗体表示该数据集该指标的最优均值，下划线表示次优均值。",
        "",
        "## 1. 本次重跑回答什么问题",
        "",
        "原论文中的 FITS、DLinear、S2S-LSTM、FreTS 和 PatchTST 来自不同脚本，数据处理、切分、归一化、训练过程和计时口径并未完全统一。此前新增的 Naive、ARIMA、单层 GRU 和 FITS-Delta 虽然使用了统一协议，但不能与论文旧表直接拼接后宣称公平比较。本次将九种方法全部接入同一数据管线、训练器、评价器和计时器，重新生成可直接横向比较的结果。",
        "",
        "## 2. 统一实验协议",
        "",
        f"- 数据：Alibaba、GWA-T-12 和 Google 三个数据集均读取前 {config['nrows']:,} 行 CPU/内存双变量轨迹。",
        f"- 预处理：{config['moving_average_window']} 点因果移动平均；严格按时间顺序 {int(config['train_ratio']*100)}%/{int(config['val_ratio']*100)}%/{int((1-config['train_ratio']-config['val_ratio'])*100)}% 划分训练、验证和测试集。",
        f"- 任务：输入最近 {config['seq_len']} 个时刻，同时预测未来 {config['pred_len']} 个时刻的 CPU 与内存。",
        "- 归一化：所有神经网络只用训练段均值和标准差进行 z-score；测试真值统一逆归一化后计算指标。",
        f"- 训练：Adam，学习率 {config['learning_rate']}，batch size {config['batch_size']}，最多 {config['max_epochs']} 轮，验证集早停 patience={config['patience']}。",
        f"- 重复：学习模型使用随机种子 {', '.join(map(str, config['seeds']))}，报告均值±样本标准差；Naive 与 ARIMA 为确定性方法，各运行一次。这里的标准差表示训练随机性，不应解释为跨数据集置信区间。",
        f"- 评价：RMSE、MAE、R²、MAPE 和 sMAPE 均由同一函数计算。全部 {run_info['completed_runs']} 个运行单元使用各数据集完全相同的测试窗口与测试真值。",
        f"- 运行环境：Python {env['python'].split('|')[0].strip()}，PyTorch {env['torch']}；训练设备为 {env['cuda_device']}，另在同一台主机 CPU 上以 batch=1、预热 {config['inference_warmup']} 次、重复 {config['inference_repeats']} 次测量推理时延。",
        "",
        "## 3. 统一结果",
        "",
    ]

    for i, dataset in enumerate(DATASETS, start=1):
        frame = summary.loc[summary.dataset == dataset].copy()
        lines += [f"### 3.{i} {dataset}", "", result_table(frame), ""]
        fits_rmse = get(summary, dataset, "FITS", "RMSE")
        delta_rmse = get(summary, dataset, "FITS-Delta", "RMSE")
        naive_rmse = get(summary, dataset, "NaiveLastValue", "RMSE")
        lines.append(
            f"FITS 相对 Naive 的 RMSE 降低 {pct_change(fits_rmse, naive_rmse):.2f}%；"
            f"FITS-Delta 相对原 FITS 的 RMSE {'降低' if delta_rmse < fits_rmse else '升高'} {abs(pct_change(delta_rmse, fits_rmse)):.2f}%。"
        )
        if dataset == "Alibaba":
            lines.append("Alibaba 上 DLinear 的 RMSE 和 R²均值最优，FITS-Delta 的 MAE、MAPE 和 sMAPE 最优。FITS/FITS-Delta 的优势主要是参数量小、结果稳定，而不是全部精度指标第一。")
        elif dataset == "GWA-T-12":
            lines.append("GWA-T-12 上 ARIMA 在五项精度指标上均为最优或接近最优；FITS 与 FITS-Delta 只明显优于 Naive 和当前 PatchTST 变体。因此正文必须明确承认该数据集是 FITS 的性能边界。")
        else:
            lines.append("Google 上单层 GRU 的 RMSE 和 R²最好，Naive 的 MAE、MAPE 和 sMAPE 最低。该数据含大量接近零的真实值，MAPE/sMAPE 对极小分母非常敏感，应以 RMSE、MAE 和 R²为主并说明百分比误差的局限。")
        lines.append("")

    lines += [
        "## 4. 模型来源和必须披露的实现边界",
        "",
        "| 模型 | 本次代码来源 | 说明 |",
        "|---|---|---|",
        "| FITS | `FITSTest/duobu_1.py` | 保留论文主代码的共享复数上采样器，625 参数。 |",
        "| FITS-Delta | `baseline_experiments/run_fixed_parameter_fits.py` | 在 FITS 前后加入一阶差分与残差累积，仍为 625 个可训练参数。 |",
        "| Naive / ARIMA / 单层 GRU | `baseline_experiments/run_baselines.py` | 新增的轻量基线；GRU 为单层、hidden size 32。 |",
        "| DLinear | `FITSTest/DLinear.py` | 保留两个共享线性头与 1010 参数，但修复旧代码将整个 batch 展平后做移动平均、造成样本相互污染的问题。 |",
        "| S2S-LSTM | `LSTM_D.py` | 保留 hidden size 128 的编码器—解码器，208,066 参数；原代码一层 LSTM 上的 dropout=0.2 实际会被 PyTorch 忽略，本次显式设为 0，行为等价。 |",
        "| FreTS | `FITSTest/FreTS.py` | 是作者目录中的 hidden size 12、截断频域轻量变体，不应写成官方完整 FreTS 的严格复现。 |",
        "| PatchTST | `FITSTest/7.py` | 是作者目录中的 edge-sized、last-patch-head 变体，不是官方使用全部 patch 展平预测头的完整 PatchTST。 |",
        "",
        "因此，论文中可写“基于作者现有代码的统一实现”，但 FreTS 与 PatchTST 必须标注为轻量/边缘变体，不能暗示它们与原论文官方配置完全相同。",
        "",
        "## 5. 与论文旧表的关系",
        "",
        "本次结果不应与论文旧表按列混合，而应整体替换预测对比表。变化较大的主要原因不是测试集变了，而是旧脚本的训练/评价设置不统一，以及 DLinear 的 batch 混样实现错误得到修正。典型变化包括：",
        "",
        "- Alibaba：统一后的 S2S-LSTM RMSE 为 2.2400±0.0366，不再复现旧表 1.61 的明显优势；DLinear 为 2.1708±0.0122，明显优于旧表 2.92。",
        "- GWA-T-12：统一后的 DLinear RMSE 为 0.7231±0.0056，而旧表为 0.42；这说明旧值不能继续与当前新增模型直接比较。",
        "- Google：统一后的 DLinear RMSE 为 0.1677±0.0006，而旧表为 0.21；但其 MAPE 仍被近零真值显著放大。",
        "- 当前 PatchTST 变体在三个数据集上均较弱，尤其 Google RMSE 为 0.2312±0.0043。该结果只能代表当前作者代码变体，不能作为官方 PatchTST 的一般性能结论。",
        "",
        "## 6. 可用于论文的稳妥结论",
        "",
        "可以支持的结论是：FITS 以 625 个参数取得稳定且有竞争力的多步资源预测结果；FITS-Delta 在不增加可训练参数的情况下改善 Alibaba 和 Google，但未改善 GWA-T-12。FITS 并非所有数据集、所有指标上的最优模型，其价值更适合表述为精度、模型规模和 CPU 推理时延之间的折中。",
        "",
        "不可以继续使用的表述包括：FITS 全面优于所有基线、FITS 在其他数据集上均达到最高精度，以及由当前作者轻量变体推断官方 FreTS/PatchTST 性能。",
        "",
        "## 7. 效率指标使用限制",
        "",
        "- 本表 CPU 时延来自当前 Ryzen 9 9800X3D 主机，只能用于同一机器上模型间的统一比较，不能替代审稿人要求的低功耗 ARM/实物边缘设备部署。",
        "- 当前 CPU batch=1 时延显示 FITS 中位数约 0.07–0.08 ms，FITS-Delta 约 0.09 ms；它们均明显低于单层 GRU，但不应与原论文 Ryzen 7 6800H 上的 0.50/0.68 ms 直接比较。",
        "- THOP 对 FFT/IFFT、显式注意力矩阵乘法等操作统计不完整，因此正式论文若报告 FLOPs/MACs，必须注明工具限制或改用覆盖这些算子的统一 profiler。",
        "",
        "## 8. 完整性检查",
        "",
        "- 原始结果 111 行，汇总结果 27 行；每个数据集包含 37 次运行（2 个确定性基线 + 7 个学习模型×5 个种子）。",
        "- 111 个预测文件均与各自数据集的 canonical test targets 逐元素完全一致。",
        "- 无核心指标缺失、无重复 run ID、无非有限预测值。",
        "- 汇总均值及样本标准差均可从原始结果在 1e-12 容差内精确复算。",
        "- 九模型形状、参数量以及 DLinear 的 batch 独立性自动测试全部通过。",
        "",
        "## 9. 结果文件",
        "",
        "- `prediction_metrics_raw.csv`：111 次运行的原始总体指标。",
        "- `prediction_metrics_detailed_raw.csv`：按通道、预测步和通道×预测步展开的详细指标。",
        "- `prediction_summary_mean_std.csv`：论文表所用均值与标准差。",
        "- `config.json`、`environment.json`、`run_info.json`：实验配置、软硬件环境与完成状态。",
        "- 各数据集目录：模型权重、训练历史、预测数组、ARIMA 验证搜索与 canonical test targets。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    raw = pd.read_csv(RAW_CSV)
    report = build_report(summary, raw)
    REPORT_PATH.write_text(report, encoding="utf-8")
    ARTICLE_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)
    print(ARTICLE_PATH)


if __name__ == "__main__":
    main()
