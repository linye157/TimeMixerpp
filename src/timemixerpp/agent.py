"""
TimeMixer++ 自主推理 Agent

将 TimeMixer++ 的三阶段推理流水线（神经网络 + 向量检索 + LLM）封装为
一个统一的、可重复使用的 Agent 类。

推理流水线：
    输入 x (48维温度序列)
        │
        ├─ y1: TimeMixerPPForBinaryCls 神经网络预测
        ├─ y2: Qdrant RAG 向量检索 + 相似度加权投票
        └─ y3: Ollama LLM 推理增强
        │
        └─> 最终概率（加权融合）+ 解释文本

典型用法：
    agent = TimeMixerAgent.from_checkpoint(
        checkpoint_path="checkpoints/best_model.pt",
        qdrant_url="http://localhost:6333",
        collection_prefix="temperature_kb",
        ollama_url="http://localhost:11434",
    )
    result = agent.predict(x)
    print(f"预测: {result.prediction}, 概率: {result.probability:.4f}")
    print(result.explanation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentResult: 结构化输出
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """
    TimeMixerAgent.predict() 的结构化输出。

    Attributes:
        probability:        最终融合概率 [0, 1]
        prediction:         二值预测结果 {0, 1}
        confidence:         预测置信度 [0, 1]（|prob - 0.5| * 2）

        y1:                 TimeMixer++ 神经网络预测概率（如启用）
        y2:                 RAG 向量投票概率（如启用）
        y3:                 LLM 推理概率（如启用）

        retrieved_samples:  各尺度检索到的相似样本字典
        llm_explanation:    LLM 推理解释文本
        llm_uncertainty:    LLM 对预测结果的不确定性 [0, 1]

        fusion_mode:        最终概率的融合方式描述
        details:            详细的中间计算结果
    """
    probability: float
    prediction: int
    confidence: float

    y1: Optional[float] = None
    y2: Optional[float] = None
    y3: Optional[float] = None

    retrieved_samples: Optional[Dict[str, List[Dict]]] = None
    llm_explanation: Optional[str] = None
    llm_uncertainty: Optional[float] = None

    fusion_mode: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为可序列化字典。"""
        return {
            "probability": round(self.probability, 4),
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "y1_timemixer": round(self.y1, 4) if self.y1 is not None else None,
            "y2_rag_vote": round(self.y2, 4) if self.y2 is not None else None,
            "y3_llm": round(self.y3, 4) if self.y3 is not None else None,
            "llm_explanation": self.llm_explanation,
            "llm_uncertainty": round(self.llm_uncertainty, 4) if self.llm_uncertainty is not None else None,
            "fusion_mode": self.fusion_mode,
            "details": self.details,
        }

    def __repr__(self) -> str:
        label_str = "事故" if self.prediction == 1 else "正常"
        parts = [
            f"TimeMixerAgent 预测结果",
            f"  结论: {label_str} (概率={self.probability:.4f}, 置信度={self.confidence:.4f})",
            f"  融合方式: {self.fusion_mode}",
        ]
        if self.y1 is not None:
            parts.append(f"  y1 (TimeMixer++): {self.y1:.4f}")
        if self.y2 is not None:
            parts.append(f"  y2 (RAG投票):     {self.y2:.4f}")
        if self.y3 is not None:
            parts.append(f"  y3 (LLM):         {self.y3:.4f}")
        if self.llm_explanation:
            parts.append(f"  LLM 解释: {self.llm_explanation[:120]}{'...' if len(self.llm_explanation) > 120 else ''}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# TimeMixerAgent
# ---------------------------------------------------------------------------

class TimeMixerAgent:
    """
    TimeMixer++ 完整推理 Agent。

    集成三个推理组件：
    - **y1**：TimeMixerPPForBinaryCls 神经网络（需要 checkpoint）
    - **y2**：Qdrant 向量数据库 RAG 检索 + 相似度加权投票
    - **y3**：Ollama 本地 LLM 推理增强

    三个组件均可独立启用/禁用，Agent 会自动降级到可用的组合。

    Args:
        model:              已初始化的 TimeMixerPPForBinaryCls 实例（可选）
        normalizer_mean:    训练集均值，用于归一化输入（可选）
        normalizer_std:     训练集标准差，用于归一化输入（可选）
        device:             推理设备（"auto" / "cpu" / "cuda"）
        qdrant_url:         Qdrant 服务地址（可选，启用 RAG）
        collection_prefix:  Qdrant collection 名称前缀（需与入库时一致）
        top_k:              每个尺度检索的相似样本数
        gamma:              相似度加权系数 exp(gamma * score)
        fusion_weights:     三尺度 RAG 投票融合权重 (w0, w1, w2)
        l2_normalize_query: 是否对查询向量 L2 归一化（需与入库时一致）
        ollama_url:         Ollama 服务地址（可选，启用 LLM）
        ollama_model:       Ollama 模型名称
        ollama_temperature: LLM 生成温度（0.0 = 确定性输出）
        llm_mode:           LLM 触发模式：
                              "none"      - 不触发
                              "always"    - 每次都触发
                              "uncertain" - 仅在概率接近阈值时触发
        threshold:          二分类阈值（默认 0.5）
        uncertain_delta:    uncertain 模式下的接近阈值范围
        provide_y1_to_llm:  是否将 y1 传递给 LLM
        provide_y2_to_llm:  是否将 y2 传递给 LLM
    """

    def __init__(
        self,
        # 模型组件
        model: Optional[Any] = None,
        normalizer_mean: Optional[np.ndarray] = None,
        normalizer_std: Optional[np.ndarray] = None,
        device: str = "auto",
        # RAG 组件
        qdrant_url: Optional[str] = None,
        collection_prefix: Optional[str] = None,
        top_k: int = 10,
        gamma: float = 10.0,
        fusion_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        l2_normalize_query: bool = False,
        # LLM 组件
        ollama_url: Optional[str] = None,
        ollama_model: str = "qwen2.5:7b",
        ollama_temperature: float = 0.0,
        llm_mode: str = "none",
        # 推理控制
        threshold: float = 0.5,
        uncertain_delta: float = 0.15,
        provide_y1_to_llm: bool = False,
        provide_y2_to_llm: bool = False,
    ):
        # --- 设备 ---
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # --- 神经网络模型 ---
        self.model = model
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        self.normalizer_mean = normalizer_mean
        self.normalizer_std = normalizer_std

        # --- RAG 组件 ---
        self.qdrant_client = None
        self.collection_prefix = collection_prefix
        self.top_k = top_k
        self.gamma = gamma
        self.fusion_weights = fusion_weights
        self.l2_normalize_query = l2_normalize_query

        if qdrant_url and collection_prefix:
            self._init_qdrant(qdrant_url)

        # --- LLM 组件 ---
        self.llm_client = None
        self.llm_mode = llm_mode
        self.provide_y1_to_llm = provide_y1_to_llm
        self.provide_y2_to_llm = provide_y2_to_llm

        if ollama_url and llm_mode != "none":
            self._init_ollama(ollama_url, ollama_model, ollama_temperature)

        # --- 推理控制 ---
        self.threshold = threshold
        self.uncertain_delta = uncertain_delta

        logger.info(
            f"TimeMixerAgent 初始化完成 | "
            f"model={'✓' if self.model else '✗'} | "
            f"rag={'✓' if self.qdrant_client else '✗'} | "
            f"llm={'✓' if self.llm_client else '✗'} | "
            f"device={self.device}"
        )

    # ------------------------------------------------------------------
    # 类方法：从 checkpoint 创建 Agent
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto",
        **kwargs,
    ) -> "TimeMixerAgent":
        """
        从 checkpoint 文件创建 Agent（最常用的创建方式）。

        Args:
            checkpoint_path: 模型 checkpoint 路径（.pt 文件）
            device:           推理设备
            **kwargs:         其他 Agent 初始化参数（RAG/LLM 配置）

        Returns:
            TimeMixerAgent 实例

        Example::

            agent = TimeMixerAgent.from_checkpoint(
                checkpoint_path="checkpoints/best_model.pt",
                qdrant_url="http://localhost:6333",
                collection_prefix="temperature_kb",
                ollama_url="http://localhost:11434",
                llm_mode="uncertain",
            )
        """
        from .config import TimeMixerPPConfig
        from .model import TimeMixerPPForBinaryCls

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
        config_dict = checkpoint.get("config", {})

        config = TimeMixerPPConfig(
            seq_len=config_dict.get("seq_len", 48),
            c_in=config_dict.get("c_in", 1),
            d_model=config_dict.get("d_model", 64),
            n_layers=config_dict.get("n_layers", 2),
            n_heads=config_dict.get("n_heads", 4),
            top_k=config_dict.get("top_k", 3),
            dropout=config_dict.get("dropout", 0.1),
        )

        model = TimeMixerPPForBinaryCls(config).to(_device)

        # 初始化 lazy 参数（模型有 lazy 初始化的组件）
        with torch.no_grad():
            dummy = torch.randn(1, config.seq_len).to(_device)
            _ = model(dummy)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        normalizer_mean = checkpoint.get("normalizer_mean")
        normalizer_std = checkpoint.get("normalizer_std")
        if normalizer_mean is not None:
            normalizer_mean = np.asarray(normalizer_mean).flatten()
        if normalizer_std is not None:
            normalizer_std = np.asarray(normalizer_std).flatten()

        logger.info(f"从 checkpoint 加载模型: {checkpoint_path}")
        return cls(
            model=model,
            normalizer_mean=normalizer_mean,
            normalizer_std=normalizer_std,
            device=device,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 初始化辅助方法
    # ------------------------------------------------------------------

    def _init_qdrant(self, qdrant_url: str) -> None:
        """尝试连接 Qdrant 服务。"""
        try:
            from .qdrant_utils import get_client
            self.qdrant_client = get_client(qdrant_url)
            logger.info(f"RAG 已连接 Qdrant: {qdrant_url}")
        except ImportError:
            logger.warning("qdrant-client 未安装，RAG 功能不可用。运行: pip install qdrant-client")
        except Exception as exc:
            logger.warning(f"连接 Qdrant 失败，RAG 功能禁用: {exc}")

    def _init_ollama(self, ollama_url: str, model: str, temperature: float) -> None:
        """尝试连接 Ollama 服务。"""
        try:
            from .ollama_client import OllamaClient
            client = OllamaClient(base_url=ollama_url, model=model, temperature=temperature)
            if client.check_connection():
                self.llm_client = client
                logger.info(f"LLM 已连接 Ollama: {ollama_url} | 模型: {model}")
            else:
                logger.warning(f"Ollama 服务不可达 ({ollama_url})，LLM 功能禁用。")
        except Exception as exc:
            logger.warning(f"初始化 Ollama 客户端失败，LLM 功能禁用: {exc}")

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def predict(
        self,
        x: Union[np.ndarray, List[float]],
        sample_id: Optional[int] = None,
        exclude_self: bool = True,
    ) -> AgentResult:
        """
        对单个样本执行完整的三阶段推理。

        Args:
            x:           48维温度时序数据（list 或 ndarray）
            sample_id:   样本全局索引（用于 RAG 排除自身）
            exclude_self: RAG 检索时是否排除自身（默认 True）

        Returns:
            AgentResult 包含最终预测及所有中间结果
        """
        x_arr = np.asarray(x, dtype=np.float32).flatten()
        if x_arr.shape[0] != 48:
            raise ValueError(f"输入必须是48维向量，实际为 {x_arr.shape[0]} 维")

        y1 = self._predict_y1(x_arr)
        ref_samples, y2, p_scales = self._predict_y2(x_arr, sample_id, exclude_self)
        y3, llm_response = self._predict_y3(x_arr, y1, y2, p_scales, ref_samples)

        return self._fuse(y1, y2, y3, ref_samples, llm_response)

    def predict_batch(
        self,
        X: Union[np.ndarray, List[List[float]]],
        sample_ids: Optional[List[int]] = None,
        exclude_self: bool = True,
    ) -> List[AgentResult]:
        """
        批量推理。

        Args:
            X:           形状 (N, 48) 的输入矩阵
            sample_ids:  每个样本的全局索引（用于 RAG 排除自身）
            exclude_self: RAG 检索时是否排除自身（默认 True）

        Returns:
            N 个 AgentResult 组成的列表
        """
        X_arr = np.asarray(X, dtype=np.float32)
        N = X_arr.shape[0]
        results = []
        for i in range(N):
            sid = sample_ids[i] if sample_ids is not None else None
            result = self.predict(X_arr[i], sample_id=sid, exclude_self=exclude_self)
            results.append(result)
            if (i + 1) % 50 == 0:
                logger.info(f"批量推理进度: {i + 1}/{N}")
        return results

    # ------------------------------------------------------------------
    # 属性：当前 Agent 功能状态
    # ------------------------------------------------------------------

    @property
    def has_model(self) -> bool:
        """是否已加载神经网络模型。"""
        return self.model is not None

    @property
    def has_rag(self) -> bool:
        """是否已连接 Qdrant RAG 服务。"""
        return self.qdrant_client is not None and self.collection_prefix is not None

    @property
    def has_llm(self) -> bool:
        """是否已连接 Ollama LLM 服务。"""
        return self.llm_client is not None and self.llm_mode != "none"

    def status(self) -> Dict[str, bool]:
        """返回各组件的可用状态。"""
        return {
            "model (y1)": self.has_model,
            "rag (y2)": self.has_rag,
            "llm (y3)": self.has_llm,
        }

    # ------------------------------------------------------------------
    # 内部方法：三阶段推理
    # ------------------------------------------------------------------

    def _predict_y1(self, x: np.ndarray) -> Optional[float]:
        """Stage 1: TimeMixer++ 神经网络预测。"""
        if not self.has_model:
            return None
        try:
            x_input = x.copy()
            if self.normalizer_mean is not None and self.normalizer_std is not None:
                x_input = (x_input - self.normalizer_mean) / (self.normalizer_std + 1e-8)

            x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(x_tensor)
            return float(output["probs"].item())
        except Exception as exc:
            logger.warning(f"神经网络预测 (y1) 失败: {exc}")
            return None

    def _predict_y2(
        self,
        x: np.ndarray,
        sample_id: Optional[int],
        exclude_self: bool,
    ) -> Tuple[Dict, Optional[float], Tuple[Optional[float], Optional[float], Optional[float]]]:
        """Stage 2: Qdrant RAG 向量检索 + 相似度加权投票。"""
        empty = ({}, None, (None, None, None))
        if not self.has_rag:
            return empty

        try:
            from .qdrant_utils import search_similar

            query_vector = x.copy()
            if self.l2_normalize_query:
                norm = np.linalg.norm(query_vector)
                if norm > 1e-8:
                    query_vector = query_vector / norm

            collection_names = [
                f"{self.collection_prefix}_scale{i}" for i in range(3)
            ]
            request_limit = self.top_k + 10 if exclude_self else self.top_k

            ref_samples: Dict[str, List[Dict]] = {}
            for scale_idx, coll_name in enumerate(collection_names):
                scale_key = f"scale{scale_idx}"
                try:
                    raw = search_similar(
                        self.qdrant_client, coll_name,
                        query_vector.tolist(), top_k=request_limit,
                        with_payload=True,
                    )
                    filtered = []
                    for r in raw:
                        sid = r.get("payload", {}).get("sample_id", r.get("id"))
                        if exclude_self and sample_id is not None and sid == sample_id:
                            continue
                        filtered.append({
                            "id": r.get("id"),
                            "sample_id": sid,
                            "score": r.get("score", 0.0),
                            "label": r.get("payload", {}).get("label", 0),
                            "label_raw": r.get("payload", {}).get(
                                "label_raw", r.get("payload", {}).get("label", 0)
                            ),
                        })
                        if len(filtered) >= self.top_k:
                            break
                    ref_samples[scale_key] = filtered
                except Exception as exc:
                    logger.warning(f"检索 {coll_name} 失败: {exc}")
                    ref_samples[scale_key] = []

            # 相似度加权投票
            w0, w1, w2 = self.fusion_weights
            scale_probs = []
            for scale_key in ["scale0", "scale1", "scale2"]:
                samples = ref_samples.get(scale_key, [])
                if not samples:
                    scale_probs.append(0.5)
                    continue
                total_w = 0.0
                weighted = 0.0
                for s in samples:
                    w = np.exp(self.gamma * s.get("score", 0.0))
                    label = s.get("label_raw", s.get("label", 0))
                    total_w += w
                    weighted += w * label
                scale_probs.append(weighted / total_w if total_w > 0 else 0.5)

            p0, p1, p2 = scale_probs
            y2 = w0 * p0 + w1 * p1 + w2 * p2
            return ref_samples, float(y2), (float(p0), float(p1), float(p2))

        except Exception as exc:
            logger.warning(f"RAG 预测 (y2) 失败: {exc}")
            return {}, None, (None, None, None)

    def _predict_y3(
        self,
        x: np.ndarray,
        y1: Optional[float],
        y2: Optional[float],
        p_scales: Tuple,
        ref_samples: Dict,
    ) -> Tuple[Optional[float], Optional[Dict]]:
        """Stage 3: Ollama LLM 推理增强。"""
        if not self.has_llm or not self._should_trigger_llm(y1, y2):
            return None, None

        try:
            from .evidence_builder import build_evidence_pack, get_valid_sample_ids
            from .ollama_client import (
                build_prediction_prompt, validate_llm_response, get_default_response
            )

            p0, p1, p2 = p_scales
            evidence = build_evidence_pack(
                query_x=x,
                ref_samples=ref_samples if ref_samples else {},
                y1=y1 if self.provide_y1_to_llm else None,
                y2=y2 if self.provide_y2_to_llm else None,
                p0=p0 if self.provide_y2_to_llm else None,
                p1=p1 if self.provide_y2_to_llm else None,
                p2=p2 if self.provide_y2_to_llm else None,
            )

            prompt = build_prediction_prompt(
                evidence,
                provide_y1=self.provide_y1_to_llm and y1 is not None,
                provide_y2=self.provide_y2_to_llm and y2 is not None,
            )

            raw_response = self.llm_client.chat(
                [{"role": "user", "content": prompt}], json_mode=True
            )

            if "error" in raw_response:
                logger.warning(f"LLM 请求失败: {raw_response['error']}")
                return None, None

            parsed = raw_response.get("parsed_json")
            valid_ids = get_valid_sample_ids(ref_samples)
            validated = validate_llm_response(
                parsed, valid_ids,
                has_y1=self.provide_y1_to_llm and y1 is not None,
                has_y2=self.provide_y2_to_llm and y2 is not None,
            )

            y3 = validated.get("y3_llm")
            return float(y3) if y3 is not None else None, validated

        except Exception as exc:
            logger.warning(f"LLM 预测 (y3) 失败: {exc}")
            return None, None

    def _should_trigger_llm(self, y1: Optional[float], y2: Optional[float]) -> bool:
        """判断是否应触发 LLM。"""
        if self.llm_mode == "none":
            return False
        if self.llm_mode == "always":
            return True
        if self.llm_mode == "uncertain":
            prob = y2 if y2 is not None else (y1 if y1 is not None else 0.5)
            return abs(prob - self.threshold) < self.uncertain_delta
        return False

    def _fuse(
        self,
        y1: Optional[float],
        y2: Optional[float],
        y3: Optional[float],
        ref_samples: Dict,
        llm_response: Optional[Dict],
    ) -> AgentResult:
        """将三个预测结果融合为最终概率。"""
        from .evidence_builder import compute_final_probability

        use_llm = llm_response is not None and y3 is not None
        final_prob, details = compute_final_probability(
            y1=y1,
            y2=y2,
            llm_response=llm_response,
            ref_samples=ref_samples,
            use_llm=use_llm,
        )

        final_prob = float(np.clip(final_prob, 0.0, 1.0))
        prediction = int(final_prob >= self.threshold)
        confidence = abs(final_prob - 0.5) * 2.0

        return AgentResult(
            probability=final_prob,
            prediction=prediction,
            confidence=confidence,
            y1=y1,
            y2=y2,
            y3=y3,
            retrieved_samples=ref_samples if ref_samples else None,
            llm_explanation=llm_response.get("explanation") if llm_response else None,
            llm_uncertainty=llm_response.get("uncertainty") if llm_response else None,
            fusion_mode=details.get("mode", "unknown"),
            details=details,
        )
