#!/usr/bin/env python
"""
TimeMixer++ Agent CLI 入口脚本

使用统一的 TimeMixerAgent 类运行完整的三阶段推理流水线：
    y1（神经网络） + y2（RAG向量检索） + y3（LLM推理）

支持：
    - 单条在线推理（--input_inline）
    - 批量文件推理（--data_path，xlsx/csv）
    - 结果输出到 CSV/JSON

Usage（示例）：

    # 最简单：仅神经网络预测
    python scripts/run_agent.py \\
        --checkpoint checkpoints/best_model.pt \\
        --input_inline "25.1,25.3,25.5,..." 

    # 添加 RAG 检索
    python scripts/run_agent.py \\
        --checkpoint checkpoints/best_model.pt \\
        --input_inline "25.1,25.3,25.5,..." \\
        --qdrant_url http://localhost:6333 \\
        --collection_prefix temperature_kb

    # 完整三阶段：神经网络 + RAG + LLM
    python scripts/run_agent.py \\
        --checkpoint checkpoints/best_model.pt \\
        --data_path data/test.csv \\
        --qdrant_url http://localhost:6333 \\
        --collection_prefix temperature_kb \\
        --ollama_url http://localhost:11434 \\
        --ollama_model qwen2.5:7b \\
        --llm_mode uncertain \\
        --output_dir results/

    # 不需要模型，仅 RAG 投票
    python scripts/run_agent.py \\
        --input_inline "25.1,25.3,..." \\
        --qdrant_url http://localhost:6333 \\
        --collection_prefix temperature_kb
"""

import sys
from pathlib import Path

# 将 src 目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import csv
import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from timemixerpp.agent import TimeMixerAgent, AgentResult


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TimeMixer++ Agent — 三阶段时序推理（神经网络 + RAG + LLM）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 输入源（二选一）──────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_inline", type=str,
        help="单条48维输入（逗号分隔数值）",
    )
    input_group.add_argument(
        "--data_path", type=str,
        help="批量输入文件路径（xlsx 或 csv）",
    )

    # ── 样本范围（批量模式）─────────────────────────────
    parser.add_argument("--start_idx", type=int, default=0,
                        help="起始样本索引（批量模式有效）")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="结束样本索引（None = 到末尾）")

    # ── y1: 神经网络模型 ────────────────────────────────
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="TimeMixer++ checkpoint 路径（启用 y1 预测）")

    # ── y2: RAG 配置 ────────────────────────────────────
    parser.add_argument("--qdrant_url", type=str, default=None,
                        help="Qdrant 服务地址（启用 RAG，示例：http://localhost:6333）")
    parser.add_argument("--collection_prefix", type=str, default=None,
                        help="Qdrant collection 前缀（与入库时保持一致）")
    parser.add_argument("--top_k", type=int, default=10,
                        help="每尺度检索的相似样本数")
    parser.add_argument("--gamma", type=float, default=10.0,
                        help="相似度加权系数 exp(gamma * score)")
    parser.add_argument("--fusion_weights", type=str, default="0.5,0.3,0.2",
                        help="三尺度融合权重（逗号分隔，三个数字之和建议为1）")
    parser.add_argument("--l2_normalize", action="store_true",
                        help="对查询向量进行 L2 归一化（需与入库时一致）")
    parser.add_argument("--exclude_self", action="store_true", default=True,
                        help="RAG 检索时排除自身样本（默认开启）")
    parser.add_argument("--no_exclude_self", dest="exclude_self", action="store_false",
                        help="RAG 检索时不排除自身")

    # ── y3: LLM 配置 ────────────────────────────────────
    parser.add_argument("--ollama_url", type=str, default=None,
                        help="Ollama 服务地址（启用 LLM，示例：http://localhost:11434）")
    parser.add_argument("--ollama_model", type=str, default="qwen2.5:7b",
                        help="Ollama 模型名称")
    parser.add_argument("--ollama_temperature", type=float, default=0.0,
                        help="LLM 生成温度（0.0=确定性输出）")
    parser.add_argument(
        "--llm_mode", type=str, default="none",
        choices=["none", "always", "uncertain"],
        help=(
            "LLM 触发模式: "
            "none=不触发, "
            "always=每次触发, "
            "uncertain=仅在预测不确定时触发"
        ),
    )
    parser.add_argument("--uncertain_delta", type=float, default=0.15,
                        help="uncertain 模式下的不确定性阈值范围")
    parser.add_argument("--provide_y1_to_llm", action="store_true",
                        help="将神经网络预测 y1 提供给 LLM")
    parser.add_argument("--provide_y2_to_llm", action="store_true",
                        help="将 RAG 投票预测 y2 提供给 LLM")

    # ── 推理控制 ─────────────────────────────────────────
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二分类决策阈值")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda")

    # ── 输出配置 ─────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default=None,
                        help="结果输出目录（None = 不保存文件，仅打印）")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")

    return parser.parse_args()


def parse_fusion_weights(s: str) -> Tuple[float, float, float]:
    """解析融合权重字符串。"""
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"fusion_weights 必须包含3个数字，实际: {s!r}")
    return (parts[0], parts[1], parts[2])


def load_input_data(args) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """加载输入数据，返回 (X, y)。y 为 None 时表示无标签。"""
    if args.input_inline:
        values = [float(v.strip()) for v in args.input_inline.split(",")]
        if len(values) != 48:
            raise ValueError(f"输入向量必须是48维，实际为 {len(values)} 维")
        return np.array([values], dtype=np.float32), None

    # 批量文件模式
    from timemixerpp.data import load_file_strict
    _, X, y = load_file_strict(args.data_path)

    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(X)
    end = min(end, len(X))

    return X[start:end].astype(np.float32), y[start:end].astype(np.float32)


def save_results(results: List[AgentResult], labels: Optional[np.ndarray],
                 output_dir: str, start_idx: int) -> None:
    """保存结果到 CSV 和 JSON。"""
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, "agent_results.csv")
    fieldnames = [
        "sample_id", "probability", "prediction", "confidence",
        "y1_timemixer", "y2_rag_vote", "y3_llm",
        "true_label", "correct", "fusion_mode", "llm_explanation",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, result in enumerate(results):
            row = result.to_dict()
            row["sample_id"] = start_idx + i
            if labels is not None:
                true_label = float(labels[i])
                row["true_label"] = round(true_label, 4)
                row["correct"] = int(result.prediction == int(true_label >= 0.5))
            else:
                row["true_label"] = None
                row["correct"] = None
            writer.writerow({k: row.get(k) for k in fieldnames})

    # JSON（详细版）
    json_path = os.path.join(output_dir, "agent_results.json")
    json_data = []
    for i, result in enumerate(results):
        item = result.to_dict()
        item["sample_id"] = start_idx + i
        if labels is not None:
            item["true_label"] = float(labels[i])
        json_data.append(item)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")


def print_summary(results: List[AgentResult], labels: Optional[np.ndarray]) -> None:
    """打印推理结果摘要。"""
    print("\n" + "=" * 60)
    print(f"推理完成 | 共 {len(results)} 个样本")
    print("=" * 60)

    if len(results) == 1:
        print(results[0])
        return

    probs = [r.probability for r in results]
    preds = [r.prediction for r in results]
    print(f"概率范围:     [{min(probs):.4f}, {max(probs):.4f}]  均值: {np.mean(probs):.4f}")
    print(f"预测分布:     正常={preds.count(0)}  事故={preds.count(1)}")

    if labels is not None:
        true_labels = (np.asarray(labels) >= 0.5).astype(int)
        correct = sum(p == t for p, t in zip(preds, true_labels))
        print(f"准确率:       {correct}/{len(results)} = {correct / len(results) * 100:.2f}%")

    fusion_modes = {}
    for r in results:
        fusion_modes[r.fusion_mode] = fusion_modes.get(r.fusion_mode, 0) + 1
    print(f"融合方式分布: {fusion_modes}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ── 解析融合权重 ────────────────────────────────────
    try:
        fusion_weights = parse_fusion_weights(args.fusion_weights)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # ── 创建 Agent ──────────────────────────────────────
    agent_kwargs = dict(
        device=args.device,
        qdrant_url=args.qdrant_url,
        collection_prefix=args.collection_prefix,
        top_k=args.top_k,
        gamma=args.gamma,
        fusion_weights=fusion_weights,
        l2_normalize_query=args.l2_normalize,
        ollama_url=args.ollama_url if args.llm_mode != "none" else None,
        ollama_model=args.ollama_model,
        ollama_temperature=args.ollama_temperature,
        llm_mode=args.llm_mode,
        uncertain_delta=args.uncertain_delta,
        threshold=args.threshold,
        provide_y1_to_llm=args.provide_y1_to_llm,
        provide_y2_to_llm=args.provide_y2_to_llm,
    )

    try:
        if args.checkpoint:
            agent = TimeMixerAgent.from_checkpoint(args.checkpoint, **agent_kwargs)
        else:
            agent = TimeMixerAgent(**agent_kwargs)
    except Exception as exc:
        logger.error(f"初始化 Agent 失败: {exc}")
        sys.exit(1)

    # 显示 Agent 状态
    status = agent.status()
    status_str = "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in status.items())
    logger.info(f"Agent 状态: {status_str}")

    if not any(status.values()):
        logger.error("所有组件均未可用（未提供 checkpoint、Qdrant 或 Ollama）。请至少配置一个推理组件。")
        sys.exit(1)

    # ── 加载数据 ────────────────────────────────────────
    try:
        X, labels = load_input_data(args)
    except Exception as exc:
        logger.error(f"加载数据失败: {exc}")
        sys.exit(1)

    N = len(X)
    start_idx = args.start_idx if args.data_path else 0
    sample_ids = list(range(start_idx, start_idx + N)) if args.exclude_self else None

    logger.info(f"开始推理 {N} 个样本...")

    # ── 执行推理 ────────────────────────────────────────
    if N == 1:
        results = [agent.predict(X[0], sample_id=sample_ids[0] if sample_ids else None,
                                 exclude_self=args.exclude_self)]
    else:
        results = agent.predict_batch(X, sample_ids=sample_ids, exclude_self=args.exclude_self)

    # ── 输出结果 ────────────────────────────────────────
    print_summary(results, labels)

    if args.output_dir:
        save_results(results, labels, args.output_dir, start_idx)


if __name__ == "__main__":
    main()
