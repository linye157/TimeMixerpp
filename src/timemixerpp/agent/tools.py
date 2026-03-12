"""
Agent 工具实现：将 TimeMixer++ 全部功能封装为可被 LLM 调用的工具。

不修改原有代码，仅通过 import 和调用封装为标准化的工具接口。

使用示例（直接调用）：

    from timemixerpp.agent import ToolRegistry
    registry = ToolRegistry()

    # 加载并预览数据
    result = registry.call("load_data", data_path="TDdata/TrainData.csv")
    print(result)  # {'n_samples': 1000, 'n_features': 48, ...}

    # 训练模型
    result = registry.call("train_model",
        data_path="TDdata/TrainData.csv", epochs=50)

    # 在测试集上评估
    result = registry.call("evaluate_model",
        checkpoint="checkpoints/best_model.pt",
        test_path="TDdata/TestData.csv")

    # 单条推理
    result = registry.call("predict",
        checkpoint="checkpoints/best_model.pt",
        input_values="25.1,25.3,...,36.8")

    # 查看模型结构
    result = registry.call("inspect_model", checkpoint="checkpoints/best_model.pt")

    # RAG 相似样本检索
    result = registry.call("rag_search",
        data_path="TDdata/TrainData.csv",
        query_index=123, top_k=10)
"""

import os
import sys
import json
import logging
import traceback
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  确保 src 和 scripts 在 sys.path 中
# ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
for p in (_SRC_DIR, _SCRIPTS_DIR, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def register_all_tools(registry):
    """注册所有内置工具到 registry。"""

    from .tool_registry import Tool

    # ================================================================
    #  1. load_data - 加载并预览数据
    # ================================================================
    def _load_data(data_path: str, preview_rows: int = 5) -> Dict[str, Any]:
        from timemixerpp.data import load_file_strict
        _, X, y = load_file_strict(data_path)
        pos_ratio = float((y >= 0.5).mean())
        return {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "label_positive_ratio": round(pos_ratio, 4),
            "label_stats": {
                "min": round(float(y.min()), 4),
                "max": round(float(y.max()), 4),
                "mean": round(float(y.mean()), 4),
            },
            "feature_stats": {
                "min": round(float(X.min()), 4),
                "max": round(float(X.max()), 4),
                "mean": round(float(X.mean()), 4),
            },
            "first_sample": X[0].tolist()[:preview_rows],
        }

    registry.add(Tool(
        name="load_data",
        description="加载 CSV/Excel 数据文件并返回基本统计信息（样本数、特征数、标签分布等）",
        parameters={
            "type": "object",
            "properties": {
                "data_path": {"type": "string", "description": "数据文件路径 (.csv 或 .xlsx)"},
                "preview_rows": {"type": "integer", "description": "预览特征的列数", "default": 5},
            },
            "required": ["data_path"],
        },
        function=_load_data,
        category="data",
    ))

    # ================================================================
    #  2. list_checkpoints - 列出已有模型检查点
    # ================================================================
    def _list_checkpoints(checkpoint_dir: str = "checkpoints") -> Dict[str, Any]:
        import torch
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.join(_PROJECT_ROOT, checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            return {"error": f"目录不存在: {checkpoint_dir}", "checkpoints": []}
        files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".pt"))
        ckpts = []
        for f in files:
            path = os.path.join(checkpoint_dir, f)
            size_mb = round(os.path.getsize(path) / 1024 / 1024, 2)
            info: Dict[str, Any] = {"file": f, "size_mb": size_mb}
            try:
                ckpt = torch.load(path, map_location="cpu")
                if "epoch" in ckpt:
                    info["epoch"] = ckpt["epoch"]
                if "metrics" in ckpt:
                    info["metrics"] = {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in ckpt["metrics"].items()
                        if k in ("accuracy", "f1", "auroc", "precision", "recall")
                    }
                if "config" in ckpt:
                    cfg = ckpt["config"]
                    info["config_summary"] = {
                        "d_model": cfg.get("d_model"),
                        "n_layers": cfg.get("n_layers"),
                        "top_k": cfg.get("top_k"),
                    }
                if "ablation" in ckpt:
                    info["ablation"] = ckpt["ablation"]
            except Exception:
                pass
            ckpts.append(info)
        return {"checkpoint_dir": checkpoint_dir, "count": len(ckpts), "checkpoints": ckpts}

    registry.add(Tool(
        name="list_checkpoints",
        description="列出 checkpoints 目录中的所有模型文件及其摘要信息",
        parameters={
            "type": "object",
            "properties": {
                "checkpoint_dir": {"type": "string", "description": "检查点目录", "default": "checkpoints"},
            },
        },
        function=_list_checkpoints,
        category="model",
    ))

    # ================================================================
    #  3. train_model - 训练 TimeMixer++ 模型
    # ================================================================
    def _train_model(
        data_path: str,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        d_model: int = 64,
        n_layers: int = 2,
        top_k: int = 3,
        ablation: str = "full",
        save_dir: str = "checkpoints",
        resume: Optional[str] = None,
        pos_weight: Optional[float] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        import subprocess
        if not os.path.isabs(data_path):
            data_path = os.path.join(_PROJECT_ROOT, data_path)
        cmd = [
            sys.executable, os.path.join(_SCRIPTS_DIR, "train.py"),
            "--data_path", data_path,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--d_model", str(d_model),
            "--n_layers", str(n_layers),
            "--top_k", str(top_k),
            "--ablation", ablation,
            "--save_dir", save_dir,
            "--seed", str(seed),
        ]
        if resume:
            cmd.extend(["--resume", resume])
        if pos_weight is not None:
            cmd.extend(["--pos_weight", str(pos_weight)])

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=3600)
        output = proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout
        return {
            "success": proc.returncode == 0,
            "return_code": proc.returncode,
            "output": output,
            "errors": proc.stderr[-1000:] if proc.stderr else "",
        }

    registry.add(Tool(
        name="train_model",
        description="训练 TimeMixer++ 二分类模型（支持消融模式、继续训练等）",
        parameters={
            "type": "object",
            "properties": {
                "data_path": {"type": "string", "description": "训练数据路径 (.csv/.xlsx)"},
                "epochs": {"type": "integer", "description": "训练轮数", "default": 50},
                "batch_size": {"type": "integer", "description": "批大小", "default": 32},
                "lr": {"type": "number", "description": "学习率", "default": 1e-3},
                "d_model": {"type": "integer", "description": "隐藏维度", "default": 64},
                "n_layers": {"type": "integer", "description": "MixerBlock 层数", "default": 2},
                "top_k": {"type": "integer", "description": "FFT Top-K 频率数", "default": 3},
                "ablation": {"type": "string", "description": "消融类型: full/no_fft/no_tid/no_mcm/no_mrm/single_scale", "default": "full"},
                "save_dir": {"type": "string", "description": "模型保存目录", "default": "checkpoints"},
                "resume": {"type": "string", "description": "恢复训练的检查点路径"},
                "pos_weight": {"type": "number", "description": "正类权重（类别不平衡时使用）"},
                "seed": {"type": "integer", "description": "随机种子", "default": 42},
            },
            "required": ["data_path"],
        },
        function=_train_model,
        category="model",
    ))

    # ================================================================
    #  4. evaluate_model - 在测试集上评估模型
    # ================================================================
    def _evaluate_model(
        checkpoint: str,
        test_path: str,
        threshold: float = 0.5,
        label_threshold: Optional[float] = None,
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        import subprocess
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(_PROJECT_ROOT, checkpoint)
        if not os.path.isabs(test_path):
            test_path = os.path.join(_PROJECT_ROOT, test_path)
        cmd = [
            sys.executable, os.path.join(_SCRIPTS_DIR, "test.py"),
            "--checkpoint", checkpoint,
            "--test_path", test_path,
            "--threshold", str(threshold),
        ]
        if label_threshold is not None:
            cmd.extend(["--label_threshold", str(label_threshold)])
        if output:
            cmd.extend(["--output", output])

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=600)
        return {
            "success": proc.returncode == 0,
            "output": proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout,
            "errors": proc.stderr[-500:] if proc.stderr else "",
        }

    registry.add(Tool(
        name="evaluate_model",
        description="在测试集上评估模型性能，输出 Accuracy/F1/AUROC/FPR/FNR 等指标",
        parameters={
            "type": "object",
            "properties": {
                "checkpoint": {"type": "string", "description": "模型检查点路径"},
                "test_path": {"type": "string", "description": "测试数据路径"},
                "threshold": {"type": "number", "description": "分类阈值", "default": 0.5},
                "label_threshold": {"type": "number", "description": "标签阈值"},
                "output": {"type": "string", "description": "预测结果保存路径"},
            },
            "required": ["checkpoint", "test_path"],
        },
        function=_evaluate_model,
        category="evaluation",
    ))

    # ================================================================
    #  5. predict - 对输入数据进行推理
    # ================================================================
    def _predict(
        checkpoint: str,
        input_values: Optional[str] = None,
        input_file: Optional[str] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        import torch
        from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
        from timemixerpp.utils import load_checkpoint

        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(_PROJECT_ROOT, checkpoint)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(checkpoint, map_location=device)
        cfg_dict = ckpt.get("config", {})
        config = TimeMixerPPConfig(**{k: v for k, v in cfg_dict.items() if hasattr(TimeMixerPPConfig, k)})
        ablation = cfg_dict.get("ablation", "full")
        model = TimeMixerPPForBinaryCls(config, ablation=ablation)

        with torch.no_grad():
            dummy = torch.randn(1, config.seq_len, device=device)
            model(dummy)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        norm_mean = ckpt.get("normalizer_mean")
        norm_std = ckpt.get("normalizer_std")

        if input_values:
            values = [float(v.strip()) for v in input_values.split(",")]
            X = np.array(values, dtype=np.float32).reshape(1, -1)
        elif input_file:
            from timemixerpp.data import load_file_strict
            if not os.path.isabs(input_file):
                input_file = os.path.join(_PROJECT_ROOT, input_file)
            _, X, _ = load_file_strict(input_file)
        else:
            return {"error": "需要提供 input_values 或 input_file"}

        if norm_mean is not None and norm_std is not None:
            X = (X - norm_mean) / (norm_std + 1e-8)

        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x_tensor)
        probs = out["probs"].cpu().numpy().flatten()
        preds = (probs >= threshold).astype(int)

        results = []
        for i in range(len(probs)):
            results.append({
                "index": i,
                "probability": round(float(probs[i]), 4),
                "prediction": int(preds[i]),
                "label": "正类(事故风险)" if preds[i] == 1 else "负类(正常)",
            })

        return {
            "n_samples": len(results),
            "threshold": threshold,
            "predictions": results if len(results) <= 20 else results[:20],
            "summary": {
                "positive_count": int(preds.sum()),
                "negative_count": int(len(preds) - preds.sum()),
                "avg_probability": round(float(probs.mean()), 4),
            },
        }

    registry.add(Tool(
        name="predict",
        description="使用训练好的模型对输入数据进行推理预测，返回事故概率",
        parameters={
            "type": "object",
            "properties": {
                "checkpoint": {"type": "string", "description": "模型检查点路径"},
                "input_values": {"type": "string", "description": "48个逗号分隔的数值（与 input_file 二选一）"},
                "input_file": {"type": "string", "description": "输入数据文件路径（与 input_values 二选一）"},
                "threshold": {"type": "number", "description": "分类阈值", "default": 0.5},
            },
            "required": ["checkpoint"],
        },
        function=_predict,
        category="model",
    ))

    # ================================================================
    #  6. inspect_model - 查看模型结构和中间形状
    # ================================================================
    def _inspect_model(
        checkpoint: Optional[str] = None,
        d_model: int = 64,
        n_layers: int = 2,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls

        if checkpoint:
            import torch
            if not os.path.isabs(checkpoint):
                checkpoint = os.path.join(_PROJECT_ROOT, checkpoint)
            ckpt = torch.load(checkpoint, map_location="cpu")
            cfg_dict = ckpt.get("config", {})
            config = TimeMixerPPConfig(**{k: v for k, v in cfg_dict.items() if hasattr(TimeMixerPPConfig, k)})
        else:
            config = TimeMixerPPConfig(d_model=d_model, n_layers=n_layers, top_k=top_k)

        M = config.compute_dynamic_M()
        scale_lengths = config.get_scale_lengths()
        model = TimeMixerPPForBinaryCls(config)
        n_params = sum(p.numel() for p in model.parameters())

        return {
            "config": {
                "seq_len": config.seq_len,
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "top_k": config.top_k,
                "dropout": config.dropout,
            },
            "dynamic_M": M,
            "scale_lengths": scale_lengths,
            "total_parameters": n_params,
            "total_parameters_readable": f"{n_params:,}",
        }

    registry.add(Tool(
        name="inspect_model",
        description="查看 TimeMixer++ 模型配置、参数量和多尺度信息",
        parameters={
            "type": "object",
            "properties": {
                "checkpoint": {"type": "string", "description": "检查点路径（可选，从中读取配置）"},
                "d_model": {"type": "integer", "description": "隐藏维度", "default": 64},
                "n_layers": {"type": "integer", "description": "层数", "default": 2},
                "top_k": {"type": "integer", "description": "Top-K", "default": 3},
            },
        },
        function=_inspect_model,
        category="model",
    ))

    # ================================================================
    #  7. extract_features - 提取多尺度特征
    # ================================================================
    def _extract_features(
        checkpoint: str,
        data_path: str,
        ablation: str = "full",
        output: Optional[str] = None,
        save_labels: bool = True,
    ) -> Dict[str, Any]:
        import subprocess
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(_PROJECT_ROOT, checkpoint)
        if not os.path.isabs(data_path):
            data_path = os.path.join(_PROJECT_ROOT, data_path)
        cmd = [
            sys.executable, os.path.join(_SCRIPTS_DIR, "extract_features.py"),
            "--checkpoint", checkpoint,
            "--data_path", data_path,
            "--ablation", ablation,
        ]
        if output:
            cmd.extend(["--output", output])
        if save_labels:
            cmd.append("--save_labels")

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=600)
        return {
            "success": proc.returncode == 0,
            "output": proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout,
            "errors": proc.stderr[-500:] if proc.stderr else "",
        }

    registry.add(Tool(
        name="extract_features",
        description="从模型提取多尺度特征并保存为 NPZ 文件",
        parameters={
            "type": "object",
            "properties": {
                "checkpoint": {"type": "string", "description": "模型检查点路径"},
                "data_path": {"type": "string", "description": "数据文件路径"},
                "ablation": {"type": "string", "description": "消融类型", "default": "full"},
                "output": {"type": "string", "description": "输出 NPZ 文件路径"},
                "save_labels": {"type": "boolean", "description": "是否保存标签", "default": True},
            },
            "required": ["checkpoint", "data_path"],
        },
        function=_extract_features,
        category="model",
    ))

    # ================================================================
    #  8. run_ablation_study - 运行消融实验
    # ================================================================
    def _run_ablation_study(
        data_path: str,
        test_path: Optional[str] = None,
        ablations: Optional[str] = None,
        epochs: int = 50,
    ) -> Dict[str, Any]:
        import subprocess
        if not os.path.isabs(data_path):
            data_path = os.path.join(_PROJECT_ROOT, data_path)
        cmd = [
            sys.executable, os.path.join(_SCRIPTS_DIR, "ablation_study.py"),
            "--data_path", data_path,
            "--epochs", str(epochs),
        ]
        if test_path:
            if not os.path.isabs(test_path):
                test_path = os.path.join(_PROJECT_ROOT, test_path)
            cmd.extend(["--test_path", test_path])
        if ablations:
            cmd.extend(["--ablations"] + ablations.split(","))

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=7200)
        return {
            "success": proc.returncode == 0,
            "output": proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout,
            "errors": proc.stderr[-500:] if proc.stderr else "",
        }

    registry.add(Tool(
        name="run_ablation_study",
        description="运行消融实验，分析各组件（TID/MCM/MRM/FFT 等）的贡献",
        parameters={
            "type": "object",
            "properties": {
                "data_path": {"type": "string", "description": "训练数据路径"},
                "test_path": {"type": "string", "description": "测试数据路径（可选）"},
                "ablations": {"type": "string", "description": "逗号分隔的消融类型列表，如 'full,no_tid,no_mcm'"},
                "epochs": {"type": "integer", "description": "训练轮数", "default": 50},
            },
            "required": ["data_path"],
        },
        function=_run_ablation_study,
        category="evaluation",
    ))

    # ================================================================
    #  9. run_baseline_comparison - 基线模型对比
    # ================================================================
    def _run_baseline_comparison(
        data_path: str,
        test_path: Optional[str] = None,
        models: Optional[str] = None,
        include_timemixer: bool = True,
        epochs: int = 50,
    ) -> Dict[str, Any]:
        import subprocess
        if not os.path.isabs(data_path):
            data_path = os.path.join(_PROJECT_ROOT, data_path)
        cmd = [
            sys.executable, os.path.join(_SCRIPTS_DIR, "baseline_comparison.py"),
            "--data_path", data_path,
            "--epochs", str(epochs),
        ]
        if test_path:
            if not os.path.isabs(test_path):
                test_path = os.path.join(_PROJECT_ROOT, test_path)
            cmd.extend(["--test_path", test_path])
        if models:
            cmd.extend(["--models"] + models.split(","))
        if include_timemixer:
            cmd.append("--include_timemixer")

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=7200)
        return {
            "success": proc.returncode == 0,
            "output": proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout,
            "errors": proc.stderr[-500:] if proc.stderr else "",
        }

    registry.add(Tool(
        name="run_baseline_comparison",
        description="与基线模型（LSTM/BiLSTM/Transformer/MLP/GRU 等）对比",
        parameters={
            "type": "object",
            "properties": {
                "data_path": {"type": "string", "description": "训练数据路径"},
                "test_path": {"type": "string", "description": "测试数据路径（可选）"},
                "models": {"type": "string", "description": "逗号分隔的模型列表，如 'lstm,transformer,mlp'"},
                "include_timemixer": {"type": "boolean", "description": "是否包含 TimeMixer++", "default": True},
                "epochs": {"type": "integer", "description": "训练轮数", "default": 50},
            },
            "required": ["data_path"],
        },
        function=_run_baseline_comparison,
        category="evaluation",
    ))

    # ================================================================
    #  10. rag_search - RAG 相似样本检索
    # ================================================================
    def _rag_search(
        data_path: str,
        query_index: int,
        top_k: int = 10,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "raw_temperature_kb",
        l2_normalize: bool = True,
        retrieve_only: bool = True,
        gamma: float = 10.0,
    ) -> Dict[str, Any]:
        import subprocess
        if not os.path.isabs(data_path):
            data_path = os.path.join(_PROJECT_ROOT, data_path)
        cmd = [
            sys.executable, os.path.join(_SCRIPTS_DIR, "query_raw_qdrant.py"),
            "--data_path", data_path,
            "--qdrant_url", qdrant_url,
            "--collection_name", collection_name,
            "--query_index", str(query_index),
            "--top_k", str(top_k),
            "--retrieve_only", str(retrieve_only).lower(),
            "--gamma", str(gamma),
            "--json_output", "true",
        ]
        if l2_normalize:
            cmd.append("--l2_normalize")

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=120)
        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                return {"success": True, "output": proc.stdout[-2000:]}
        return {"success": False, "errors": proc.stderr[-500:]}

    registry.add(Tool(
        name="rag_search",
        description="在 Qdrant 知识库中检索相似温度序列样本",
        parameters={
            "type": "object",
            "properties": {
                "data_path": {"type": "string", "description": "包含查询样本的数据文件"},
                "query_index": {"type": "integer", "description": "查询样本索引"},
                "top_k": {"type": "integer", "description": "返回的相似样本数", "default": 10},
                "qdrant_url": {"type": "string", "description": "Qdrant 地址", "default": "http://localhost:6333"},
                "collection_name": {"type": "string", "description": "Collection 名称", "default": "raw_temperature_kb"},
                "l2_normalize": {"type": "boolean", "description": "是否 L2 归一化", "default": True},
                "retrieve_only": {"type": "boolean", "description": "仅检索(true)还是检索+预测(false)", "default": True},
                "gamma": {"type": "number", "description": "相似度加权系数", "default": 10.0},
            },
            "required": ["data_path", "query_index"],
        },
        function=_rag_search,
        category="rag",
    ))

    # ================================================================
    #  11. rag_predict - RAG + LLM 综合预测
    # ================================================================
    def _rag_predict(
        input_inline: Optional[str] = None,
        data_path: Optional[str] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        qdrant_url: str = "http://localhost:6333",
        collection_prefix: str = "raw_temperature_kb",
        use_y1: bool = False,
        use_y2: bool = True,
        timemixer_ckpt: Optional[str] = None,
        llm_mode: str = "none",
        ollama_model: str = "qwen2.5:7b",
        l2_normalize: bool = True,
    ) -> Dict[str, Any]:
        import subprocess
        cmd = [sys.executable, os.path.join(_SCRIPTS_DIR, "predict_with_ollama_rag.py")]
        if input_inline:
            cmd.extend(["--input_inline", input_inline])
        elif data_path:
            if not os.path.isabs(data_path):
                data_path = os.path.join(_PROJECT_ROOT, data_path)
            cmd.extend(["--data_path", data_path, "--start_idx", str(start_idx)])
            if end_idx is not None:
                cmd.extend(["--end_idx", str(end_idx)])
        else:
            return {"error": "需要提供 input_inline 或 data_path"}

        cmd.extend([
            "--qdrant_url", qdrant_url,
            "--collection_prefix", collection_prefix,
            "--use_y1", str(use_y1).lower(),
            "--use_y2", str(use_y2).lower(),
            "--llm_mode", llm_mode,
            "--ollama_model", ollama_model,
            "--user_confirm", "false",
        ])
        if l2_normalize:
            cmd.append("--l2_normalize")
        if timemixer_ckpt:
            cmd.extend(["--timemixer_ckpt", timemixer_ckpt])

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=1800)
        return {
            "success": proc.returncode == 0,
            "output": proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout,
            "errors": proc.stderr[-500:] if proc.stderr else "",
        }

    registry.add(Tool(
        name="rag_predict",
        description="RAG 检索 + 可选 LLM 解释的综合预测（支持单条/批量输入）",
        parameters={
            "type": "object",
            "properties": {
                "input_inline": {"type": "string", "description": "48个逗号分隔的数值"},
                "data_path": {"type": "string", "description": "批量输入文件路径"},
                "start_idx": {"type": "integer", "description": "起始样本索引", "default": 0},
                "end_idx": {"type": "integer", "description": "结束样本索引"},
                "qdrant_url": {"type": "string", "description": "Qdrant 地址", "default": "http://localhost:6333"},
                "collection_prefix": {"type": "string", "description": "Collection 前缀"},
                "use_y1": {"type": "boolean", "description": "是否使用 TimeMixer++ 预测", "default": False},
                "use_y2": {"type": "boolean", "description": "是否使用 RAG 投票", "default": True},
                "timemixer_ckpt": {"type": "string", "description": "TimeMixer++ 检查点路径"},
                "llm_mode": {"type": "string", "description": "LLM 模式: none/top/uncertain/all", "default": "none"},
                "ollama_model": {"type": "string", "description": "Ollama 模型", "default": "qwen2.5:7b"},
                "l2_normalize": {"type": "boolean", "description": "L2 归一化", "default": True},
            },
        },
        function=_rag_predict,
        category="rag",
    ))

    # ================================================================
    #  12. compute_metrics - 计算评估指标
    # ================================================================
    def _compute_metrics(
        y_true: str,
        y_pred: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        from timemixerpp.utils import compute_metrics as _cm
        true_arr = np.array([float(v.strip()) for v in y_true.split(",")])
        pred_arr = np.array([float(v.strip()) for v in y_pred.split(",")])
        metrics = _cm(true_arr, pred_arr, threshold=threshold)
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

    registry.add(Tool(
        name="compute_metrics",
        description="给定真实标签和预测概率，计算分类指标（Acc/F1/AUROC/FPR/FNR）",
        parameters={
            "type": "object",
            "properties": {
                "y_true": {"type": "string", "description": "逗号分隔的真实标签"},
                "y_pred": {"type": "string", "description": "逗号分隔的预测概率"},
                "threshold": {"type": "number", "description": "分类阈值", "default": 0.5},
            },
            "required": ["y_true", "y_pred"],
        },
        function=_compute_metrics,
        category="evaluation",
    ))

    # ================================================================
    #  13. list_files - 列出项目文件
    # ================================================================
    def _list_files(directory: str = ".", pattern: str = "*") -> Dict[str, Any]:
        import glob
        target = os.path.join(_PROJECT_ROOT, directory)
        if not os.path.isdir(target):
            return {"error": f"目录不存在: {directory}"}
        matches = sorted(glob.glob(os.path.join(target, pattern)))
        items = []
        for p in matches[:100]:
            rel = os.path.relpath(p, _PROJECT_ROOT)
            is_dir = os.path.isdir(p)
            items.append({"path": rel, "is_dir": is_dir})
        return {"directory": directory, "count": len(items), "items": items}

    registry.add(Tool(
        name="list_files",
        description="列出项目目录下的文件和文件夹",
        parameters={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "目录路径（相对于项目根目录）", "default": "."},
                "pattern": {"type": "string", "description": "文件匹配模式", "default": "*"},
            },
        },
        function=_list_files,
        category="system",
    ))

    # ================================================================
    #  14. get_help - 获取帮助信息
    # ================================================================
    def _get_help(topic: str = "overview") -> Dict[str, Any]:
        topics = {
            "overview": (
                "TimeMixer++ 是一个用于时间序列二分类（事故概率预测）的深度学习框架。"
                "核心组件：MRTI（多分辨率时间成像）、TID（时间图像分解）、MCM（多尺度混合）、MRM（多分辨率混合）。"
                "支持功能：模型训练/测试/推理、消融实验、基线对比、RAG 检索、LLM 增强推理。"
            ),
            "data_format": (
                "CSV: 无表头，0-47列为48个温度特征，48列为标签。"
                "Excel: Sheet3，4-51列为特征，52列为标签。"
                "标签为 0-1 之间的浮点数。"
            ),
            "workflow": (
                "典型工作流：1) load_data 查看数据 → 2) train_model 训练 → "
                "3) evaluate_model 评估 → 4) predict 推理。"
                "进阶流程：extract_features → RAG 入库 → rag_search / rag_predict。"
            ),
            "ablation": (
                "可用消融类型：full(完整模型), no_fft, no_tid, no_mcm, no_mrm, single_scale, "
                "top_k_1, top_k_5, layers_1, layers_4, d_model_32, d_model_128。"
            ),
        }
        content = topics.get(topic, f"未知主题: {topic}。可选: {', '.join(topics.keys())}")
        return {"topic": topic, "content": content}

    registry.add(Tool(
        name="get_help",
        description="获取项目帮助信息（overview/data_format/workflow/ablation）",
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "帮助主题: overview/data_format/workflow/ablation",
                    "default": "overview",
                    "enum": ["overview", "data_format", "workflow", "ablation"],
                },
            },
        },
        function=_get_help,
        category="system",
    ))

    logger.info(f"Registered {len(registry.list_names())} tools")
