"""
TimeMixer++ Agent 框架

提供 ReAct 和 FunctionCall 两种 Agent 模式，将 TimeMixer++ 的全部功能
封装为可被 LLM 调用的工具函数。

使用示例：

    # 1. ReAct 模式（交互式推理）
    from timemixerpp.agent import ReActAgent, ToolRegistry

    registry = ToolRegistry()
    agent = ReActAgent(registry, ollama_model="qwen2.5:7b")
    result = agent.run("用 TDdata/TrainData.csv 训练模型，然后在测试集上评估")

    # 2. FunctionCall 模式（程序化调用）
    from timemixerpp.agent import ToolRegistry

    registry = ToolRegistry()
    result = registry.call("load_data", data_path="TDdata/TrainData.csv")
    print(result)

    # 3. 获取 OpenAI 兼容的工具 schema
    schemas = registry.get_openai_tools()
"""

from .tool_registry import ToolRegistry, Tool
from .tools import register_all_tools
from .react_agent import ReActAgent

__all__ = [
    "ToolRegistry",
    "Tool",
    "ReActAgent",
    "register_all_tools",
]
