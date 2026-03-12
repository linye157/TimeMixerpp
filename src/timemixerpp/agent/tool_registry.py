"""
工具注册表：管理所有可被 Agent 调用的工具。

每个工具包含：
- name: 工具唯一标识
- description: 工具功能描述
- parameters: JSON Schema 格式的参数定义
- function: 实际执行函数

支持输出为 OpenAI Function Calling 兼容格式。

使用示例：

    from timemixerpp.agent.tool_registry import ToolRegistry, Tool

    registry = ToolRegistry()

    # 注册工具
    @registry.register(
        name="my_tool",
        description="这是一个示例工具",
        parameters={
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "参数1"}
            },
            "required": ["arg1"]
        }
    )
    def my_tool(arg1: str) -> dict:
        return {"result": arg1}

    # 调用工具
    result = registry.call("my_tool", arg1="hello")

    # 获取 OpenAI 兼容 schema
    tools = registry.get_openai_tools()
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """工具定义。"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable[..., Any]
    category: str = "general"

    def to_openai_schema(self) -> Dict[str, Any]:
        """转为 OpenAI function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_summary(self) -> str:
        """简短描述，供 ReAct Agent 参考。"""
        params = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        param_strs = []
        for pname, pdef in params.items():
            req = "*" if pname in required else ""
            param_strs.append(f"{pname}{req}: {pdef.get('type', 'any')}")
        params_text = ", ".join(param_strs) if param_strs else "无参数"
        return f"{self.name}({params_text}) - {self.description}"


class ToolRegistry:
    """
    工具注册表。

    集中管理所有工具，支持注册、查找、调用和 schema 导出。
    实例化时自动注册所有内置工具。
    """

    def __init__(self, auto_register: bool = True):
        self._tools: Dict[str, Tool] = {}
        if auto_register:
            from .tools import register_all_tools
            register_all_tools(self)

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        category: str = "general",
    ) -> Callable:
        """装饰器：注册工具函数。"""
        def decorator(func: Callable) -> Callable:
            tool = Tool(
                name=name,
                description=description,
                parameters=parameters,
                function=func,
                category=category,
            )
            self._tools[name] = tool
            logger.debug(f"Registered tool: {name}")
            return func
        return decorator

    def add(self, tool: Tool):
        """直接添加 Tool 对象。"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[Tool]:
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def call(self, name: str, **kwargs) -> Any:
        """
        按名称调用工具。

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具返回值

        Raises:
            KeyError: 工具不存在
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys())
            raise KeyError(f"Tool '{name}' not found. Available: {available}")
        try:
            result = tool.function(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            return {"error": str(e), "tool": name}

    def call_from_json(self, name: str, arguments_json: str) -> Any:
        """从 JSON 字符串参数调用工具（适配 OpenAI function_call 格式）。"""
        kwargs = json.loads(arguments_json) if arguments_json else {}
        return self.call(name, **kwargs)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """导出所有工具为 OpenAI function calling 兼容格式。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_tools_prompt(self) -> str:
        """生成供 ReAct Agent 使用的工具列表描述。"""
        lines = ["可用工具列表：", ""]
        categories: Dict[str, List[Tool]] = {}
        for tool in self._tools.values():
            categories.setdefault(tool.category, []).append(tool)

        category_names = {
            "data": "数据管理",
            "model": "模型操作",
            "evaluation": "评估分析",
            "rag": "RAG 检索",
            "system": "系统工具",
            "general": "通用",
        }

        for cat, tools in sorted(categories.items()):
            cat_display = category_names.get(cat, cat)
            lines.append(f"## {cat_display}")
            for t in tools:
                lines.append(f"  - {t.to_summary()}")
            lines.append("")

        return "\n".join(lines)
