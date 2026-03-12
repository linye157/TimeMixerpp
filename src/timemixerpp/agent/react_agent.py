"""
ReAct Agent：基于 Thought → Action → Observation 循环的自主推理 Agent。

通过 Ollama 本地 LLM 驱动，将用户自然语言指令转化为工具调用序列。
同时支持 FunctionCall（OpenAI 格式）作为底层调用协议。

使用示例：

    from timemixerpp.agent import ReActAgent, ToolRegistry

    # 方式 1: 使用 Ollama 本地 LLM
    registry = ToolRegistry()
    agent = ReActAgent(
        registry,
        ollama_url="http://localhost:11434",
        ollama_model="qwen2.5:7b",
    )
    result = agent.run("加载 TDdata/TrainData.csv 看看数据情况")
    print(result.final_answer)

    # 方式 2: 单步手动调用（不需要 LLM）
    registry = ToolRegistry()
    output = registry.call("load_data", data_path="TDdata/TrainData.csv")
    print(output)

    # 方式 3: 交互式 CLI
    agent.interactive()
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  数据结构
# ──────────────────────────────────────────────

@dataclass
class AgentStep:
    """Agent 单步记录。"""
    step_num: int
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    error: str = ""


@dataclass
class AgentResult:
    """Agent 运行结果。"""
    query: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    success: bool = True
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "steps": [
                {
                    "step": s.step_num,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation[:500],
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "success": self.success,
            "total_time": round(self.total_time, 2),
        }


# ──────────────────────────────────────────────
#  System Prompt 模板
# ──────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """你是 TimeMixer++ Agent，一个智能的时间序列分析助手。
你可以通过调用工具来完成用户的请求。

## 工作方式

你按照 ReAct 模式工作：每一步先思考（Thought），然后选择工具执行（Action），再根据结果继续推理。

## 输出格式

每一步必须输出以下格式的 JSON：

{{
  "thought": "我需要做什么，为什么",
  "action": "工具名称",
  "action_input": {{"参数名": "参数值"}},
}}

当你认为任务已完成，输出：

{{
  "thought": "任务完成，总结结论",
  "action": "finish",
  "action_input": {{"answer": "最终回答"}}
}}

## 重要规则

1. 每次只输出一个 JSON 对象
2. action 必须是下面列出的工具名称之一，或 "finish"
3. action_input 必须是合法的 JSON 对象，key 与工具参数名一致
4. 不要输出 JSON 以外的内容
5. 路径使用相对路径（相对于项目根目录），如 "TDdata/TrainData.csv"

{tools_prompt}
"""


# ──────────────────────────────────────────────
#  ReAct Agent
# ──────────────────────────────────────────────

class ReActAgent:
    """
    ReAct Agent：通过 LLM 驱动的自主推理循环。

    支持三种运行模式：
    1. run(query) - 自动执行直到完成
    2. step(query) - 单步执行（手动控制循环）
    3. interactive() - CLI 交互模式
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b",
        max_steps: int = 15,
        temperature: float = 0.0,
        verbose: bool = True,
    ):
        self.registry = registry or ToolRegistry()
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.max_steps = max_steps
        self.temperature = temperature
        self.verbose = verbose

        self._system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_prompt=self.registry.get_tools_prompt()
        )
        self._history: List[Dict[str, str]] = []

    # ─────── 核心运行方法 ───────

    def run(self, query: str) -> AgentResult:
        """
        运行 Agent 直到任务完成或达到最大步数。

        Args:
            query: 用户问题或指令

        Returns:
            AgentResult 包含所有步骤和最终回答
        """
        start = time.time()
        result = AgentResult(query=query)

        self._history = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]

        for step_num in range(1, self.max_steps + 1):
            step = AgentStep(step_num=step_num)

            llm_output = self._call_llm()
            parsed = self._parse_response(llm_output)

            if parsed is None:
                step.thought = "LLM 输出解析失败"
                step.error = llm_output[:500]
                result.steps.append(step)
                self._history.append({"role": "assistant", "content": llm_output})
                self._history.append({
                    "role": "user",
                    "content": "输出格式错误，请严格按照 JSON 格式输出。"
                })
                continue

            step.thought = parsed.get("thought", "")
            step.action = parsed.get("action", "")
            step.action_input = parsed.get("action_input", {})

            self._history.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"步骤 {step_num}")
                print(f"思考: {step.thought}")
                print(f"动作: {step.action}")
                print(f"参数: {json.dumps(step.action_input, ensure_ascii=False, indent=2)}")

            if step.action == "finish":
                result.final_answer = step.action_input.get("answer", step.thought)
                result.steps.append(step)
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"最终回答: {result.final_answer}")
                break

            observation = self._execute_tool(step.action, step.action_input)
            step.observation = observation

            if self.verbose:
                obs_preview = observation[:800] + "..." if len(observation) > 800 else observation
                print(f"观察: {obs_preview}")

            self._history.append({"role": "user", "content": f"Observation:\n{observation}"})
            result.steps.append(step)
        else:
            result.final_answer = "达到最大步数限制，任务未完成。"
            result.success = False

        result.total_time = time.time() - start
        return result

    def run_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """
        FunctionCall 模式：直接执行指定的工具调用。

        兼容 OpenAI function_call 协议。

        Args:
            function_name: 工具名称
            arguments: 工具参数字典

        Returns:
            工具执行结果
        """
        return self.registry.call(function_name, **arguments)

    # ─────── 交互式 CLI ───────

    def interactive(self):
        """启动交互式命令行界面。"""
        print("=" * 60)
        print(" TimeMixer++ Agent 交互模式")
        print(" 输入自然语言指令，Agent 将自动调用工具完成任务")
        print(" 输入 'quit' 退出, 'tools' 查看可用工具")
        print("=" * 60)

        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("再见！")
                break
            if query.lower() == "tools":
                print(self.registry.get_tools_prompt())
                continue
            if query.lower() == "help":
                print("输入自然语言指令，如：")
                print("  '加载 TDdata/TrainData.csv 看看数据'")
                print("  '列出所有检查点'")
                print("  '用训练数据训练一个模型'")
                continue

            result = self.run(query)
            if not result.success:
                print(f"\n[警告] 任务未完成: {result.final_answer}")

    # ─────── 内部方法 ───────

    def _call_llm(self) -> str:
        """调用 Ollama LLM。"""
        from timemixerpp.ollama_client import OllamaClient

        client = OllamaClient(
            base_url=self.ollama_url,
            model=self.ollama_model,
            temperature=self.temperature,
        )

        response = client.chat(messages=self._history, json_mode=True)

        if "error" in response:
            logger.error(f"LLM call failed: {response['error']}")
            return json.dumps({
                "thought": f"LLM 调用失败: {response['error']}",
                "action": "finish",
                "action_input": {"answer": f"错误: {response['error']}"}
            })

        return response.get("content", "")

    def _parse_response(self, content: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 输出为结构化 dict。"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _execute_tool(self, action: str, action_input: Dict[str, Any]) -> str:
        """执行工具并返回观察结果字符串。"""
        try:
            result = self.registry.call(action, **action_input)
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False, indent=2, default=str)
            return str(result)
        except KeyError as e:
            return f"错误: 工具 '{action}' 不存在。可用工具: {', '.join(self.registry.list_names())}"
        except Exception as e:
            return f"工具执行异常: {type(e).__name__}: {e}"

    # ─────── 导出方法 ───────

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """导出 OpenAI function calling 格式的工具列表。"""
        return self.registry.get_openai_tools()

    def get_tools_summary(self) -> str:
        """获取工具列表摘要。"""
        return self.registry.get_tools_prompt()
