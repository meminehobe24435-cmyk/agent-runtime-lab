"""命令行入口：``python -m agentlab <command>``。

四个子命令，覆盖「看工具 → 跑单个任务 → 跑评测 → 跑多智能体」：

- ``tools``   打印工具清单（含自动生成的 JSON Schema）
- ``run``     用规则后端跑一个任务，打印 trace
- ``eval``    跑内置 12 任务评测集，产出 JSON + Markdown 报告
- ``demo``    跑多智能体协作演示（Supervisor + 3 个 Worker）
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from .eval import (
    build_knowledge_base,
    build_ops_tools,
    make_scenario_factory,
    run_default_suite,
    scenario_summary,
)
from .llm import RuleBasedBackend
from .orchestrator import Supervisor, Worker
from .runtime import AgentConfig, AgentRuntime
from .tools import ToolRegistry, calculator


def _force_utf8() -> None:
    """Windows 控制台默认是 GBK，打印中文/箭头会抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #

def cmd_tools(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        kb = build_knowledge_base()
        registry = build_ops_tools(kb, Path(tmp))
        print(f"共 {len(registry)} 个工具：")
        for spec in registry:
            print(f"\n- {spec.name}  [{', '.join(spec.tags) or '未分类'}]")
            print(f"  {spec.description}")
            print("  " + json.dumps(spec.parameters, ensure_ascii=False))
        print("\n" + json.dumps(scenario_summary(registry, kb), ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        kb = build_knowledge_base()
        registry = build_ops_tools(kb, Path(tmp))
        rules = [
            {"match": ("温度", "cnc-03"), "tool": "query_metric",
             "arguments": {"device": "cnc-03", "metric": "temperature", "window_min": 10},
             "until": "observed", "answer": "cnc-03 温度末值 62.8 ℃，处于上升通道。"},
            {"match": ("规程",), "tool": "search_knowledge_base",
             "arguments": {"query": "温升异常 处置规程", "k": 3},
             "until": "observed", "answer": "已检索处置规程，按 sop-overheat 执行。"},
        ]
        backend = RuleBasedBackend(rules)
        agent = AgentRuntime(backend, registry, config=AgentConfig(max_steps=6),
                             knowledge=kb, name="cli-agent")
        result = agent.run(args.task)
        print(result.tracer.render() if result.tracer else "")
        print("\n== 最终回答 ==")
        print(result.answer)
        print("\n== 结果 ==")
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


def cmd_eval(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sandbox = out_dir / "sandbox"

    report = run_default_suite(sandbox)
    (out_dir / "eval_report.json").write_text(report.to_json(), encoding="utf-8")
    (out_dir / "eval_report.md").write_text(report.to_markdown(), encoding="utf-8")

    metrics = report.metrics()
    print(f"评测集：{report.label}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("\n逐任务：")
    for outcome in report.outcomes:
        flag = "PASS" if outcome.passed else "FAIL"
        failed = ("  失败项: " + "、".join(outcome.failed_checks)) if outcome.failed_checks else ""
        print(f"  [{flag}] {outcome.task.id:<28} 步数={outcome.result.steps if outcome.result else 0}"
              f" 修正={outcome.result.corrections if outcome.result else 0}{failed}")
    print(f"\n报告已写入: {out_dir}")
    return 0 if report.passed == report.total else 1


def cmd_demo(args: argparse.Namespace) -> int:
    """多智能体演示：一个取数 Worker + 一个规程 Worker + 一个计算 Worker。"""
    with tempfile.TemporaryDirectory() as tmp:
        kb = build_knowledge_base()
        registry = build_ops_tools(kb, Path(tmp))

        def worker(name: str, skills: List[str], tools: List[str], rules: List[dict]) -> Worker:
            sub = ToolRegistry([calculator] if "calculator" in tools else [])
            for tool_name in tools:
                if tool_name in registry:
                    sub.register(registry.get(tool_name))
            runtime = AgentRuntime(
                RuleBasedBackend(rules), sub, config=AgentConfig(max_steps=4),
                knowledge=kb, name=name,
            )
            return Worker(name=name, runtime=runtime, skills=skills)

        collector = worker(
            "采集员", ["指标", "温度", "电流", "振动"],
            ["query_metric", "flaky_gateway"],
            [{"match": ("指标", "温度"), "tool": "query_metric",
              "arguments": {"device": "cnc-03", "metric": "temperature", "window_min": 10},
              "until": "observed", "answer": "cnc-03 温度序列末值 62.8 ℃。"},
             {"match": ("心跳",), "tool": "flaky_gateway",
              "arguments": {"device": "pump-07"}, "until": "observed",
              "answer": "pump-07 心跳正常。"}],
        )
        researcher = worker(
            "规程员", ["规程", "处置", "异常"],
            ["search_knowledge_base"],
            [{"match": ("规程",), "tool": "search_knowledge_base",
              "arguments": {"query": "温升异常 处置", "k": 3}, "until": "observed",
              "answer": "检索到 sop-overheat：转速降至 60%，检查冷却液流量。"}],
        )
        calculator_worker = worker(
            "计算员", ["计算", "余量", "换算"],
            ["calculator", "convert_unit"],
            [{"match": ("计算", "余量"), "tool": "calculator",
              "arguments": {"expression": "(85-68.4)/85*100"}, "until": "observed",
              "answer": "散热余量约 19.5%。"}],
        )

        supervisor = Supervisor(
            [collector, researcher, calculator_worker],
            bus=None, synthesized=False, max_parts=3,
        )
        result = supervisor.run(args.task)

        print("== 子任务分配与结果 ==")
        for sub in result.subtasks:
            print(f"[{sub.task_id}] {sub.status:<6} 负责人={sub.assignee or '-':<8} {sub.description}")
            if sub.answer:
                print(f"      → {sub.answer}")
            if sub.error:
                print(f"      ! {sub.error}")
        print("\n== 消息总线 ==")
        for msg in result.bus.history if result.bus else []:
            print(f"  {msg.sender} -> {msg.recipient} [{msg.kind}] "
                  f"{json.dumps(msg.payload, ensure_ascii=False)[:110]}")
        print("\n== 汇总回答 ==")
        print(result.answer)
        print(f"\n成功={result.success} 原因={result.reason} 轮次={result.rounds} "
              f"耗时={result.duration_ms:.1f}ms")
    return 0 if result.success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentlab",
        description="零依赖 LLM Agent 运行时（工具调用 / 记忆 / RAG / 多智能体 / 评测）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tools = sub.add_parser("tools", help="打印工具清单与 JSON Schema")
    p_tools.set_defaults(func=cmd_tools)

    p_run = sub.add_parser("run", help="用规则后端跑一个任务并打印 trace")
    p_run.add_argument("task", nargs="?", default="查一下 cnc-03 的温度指标")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="跑内置评测集并产出报告")
    p_eval.add_argument("--out", default="reports", help="报告输出目录（默认 reports）")
    p_eval.set_defaults(func=cmd_eval)

    p_demo = sub.add_parser("demo", help="多智能体协作演示")
    p_demo.add_argument(
        "task", nargs="?",
        default="查一下 cnc-03 的温度指标，然后检索温升异常规程，最后计算散热余量",
    )
    p_demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
