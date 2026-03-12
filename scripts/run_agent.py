"""
TimeMixer++ Agent 启动脚本

支持三种运行模式：
  1. interactive  - 交互式 CLI，与 Agent 对话
  2. run          - 执行单条指令
  3. call         - 直接调用工具（FunctionCall 模式）

使用示例：

    # 交互模式（需要 Ollama 服务运行中）
    python scripts/run_agent.py interactive --model qwen2.5:7b

    # 执行单条指令
    python scripts/run_agent.py run "加载 TDdata/TrainData.csv 看看数据"

    # 直接调用工具（FunctionCall 模式，不需要 LLM）
    python scripts/run_agent.py call load_data --args '{"data_path": "TDdata/TrainData.csv"}'
    python scripts/run_agent.py call list_checkpoints
    python scripts/run_agent.py call predict --args '{"checkpoint": "checkpoints/best_model.pt", "input_values": "25.1,25.3,...,36.8"}'

    # 列出所有可用工具
    python scripts/run_agent.py tools

    # 导出 OpenAI FunctionCall 格式 schema
    python scripts/run_agent.py schema --output tools_schema.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(
        description="TimeMixer++ Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s interactive                          # 交互模式
  %(prog)s run "查看训练数据统计"                # 单条指令
  %(prog)s call load_data --args '{"data_path":"TDdata/TrainData.csv"}'  # 工具调用
  %(prog)s tools                                # 列出工具
  %(prog)s schema                               # 导出 schema
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # interactive
    p_inter = subparsers.add_parser("interactive", help="交互式 Agent CLI")
    p_inter.add_argument("--model", default="qwen2.5:7b", help="Ollama 模型名称")
    p_inter.add_argument("--ollama_url", default="http://localhost:11434", help="Ollama 地址")
    p_inter.add_argument("--max_steps", type=int, default=15, help="最大推理步数")

    # run
    p_run = subparsers.add_parser("run", help="执行单条自然语言指令")
    p_run.add_argument("query", type=str, help="要执行的指令")
    p_run.add_argument("--model", default="qwen2.5:7b", help="Ollama 模型名称")
    p_run.add_argument("--ollama_url", default="http://localhost:11434", help="Ollama 地址")
    p_run.add_argument("--max_steps", type=int, default=15, help="最大推理步数")
    p_run.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")

    # call (FunctionCall)
    p_call = subparsers.add_parser("call", help="直接调用工具（FunctionCall 模式）")
    p_call.add_argument("tool_name", type=str, help="工具名称")
    p_call.add_argument("--args", type=str, default="{}", help="JSON 格式的参数")

    # tools
    subparsers.add_parser("tools", help="列出所有可用工具")

    # schema
    p_schema = subparsers.add_parser("schema", help="导出 OpenAI FunctionCall schema")
    p_schema.add_argument("--output", type=str, default=None, help="输出文件路径")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    from timemixerpp.agent import ToolRegistry, ReActAgent

    if args.mode == "tools":
        registry = ToolRegistry()
        print(registry.get_tools_prompt())
        return

    if args.mode == "schema":
        registry = ToolRegistry()
        schema = registry.get_openai_tools()
        output = json.dumps(schema, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Schema 已保存到: {args.output}")
        else:
            print(output)
        return

    if args.mode == "call":
        registry = ToolRegistry()
        try:
            kwargs = json.loads(args.args)
        except json.JSONDecodeError as e:
            print(f"参数 JSON 解析失败: {e}")
            sys.exit(1)
        result = registry.call(args.tool_name, **kwargs)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    if args.mode == "interactive":
        agent = ReActAgent(
            ollama_url=args.ollama_url,
            ollama_model=args.model,
            max_steps=args.max_steps,
        )
        agent.interactive()
        return

    if args.mode == "run":
        agent = ReActAgent(
            ollama_url=args.ollama_url,
            ollama_model=args.model,
            max_steps=args.max_steps,
        )
        result = agent.run(args.query)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"最终回答: {result.final_answer}")
            print(f"总耗时: {result.total_time:.1f}s, 步骤数: {len(result.steps)}")


if __name__ == "__main__":
    main()
