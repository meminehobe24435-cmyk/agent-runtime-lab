"""多智能体测试：任务拆分、Worker 选择、失败改派、消息总线、汇总口径。"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agentlab.llm import ScriptedBackend
from agentlab.orchestrator import (
    BusMessage,
    MessageBus,
    SubTask,
    Supervisor,
    Worker,
    default_decompose,
)
from agentlab.runtime import AgentConfig, AgentRuntime
from agentlab.tools import ToolRegistry
from agentlab.trace import PHASE_DELEGATE, PHASE_ERROR


# --------------------------------------------------------------------------- #
# 测试用工具与 Worker
# --------------------------------------------------------------------------- #

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def query_metric(device: str, metric: str = "temperature") -> str:
        """查指标。

        Args:
            device: 设备。
            metric: 指标。
        """
        return f"{device}:{metric}=62.8"

    def search_knowledge_base(query: str) -> str:
        """查规程。

        Args:
            query: 查询。
        """
        return f"规程命中: {query}"

    def calculator(expression: str) -> str:
        """算数。

        Args:
            expression: 表达式。
        """
        return "19.5"

    def boom(x: int = 1) -> str:
        """总失败。

        Args:
            x: 参数。
        """
        raise RuntimeError("工具不可用")

    registry.register_function(query_metric, name="query_metric")
    registry.register_function(search_knowledge_base, name="search_knowledge_base")
    registry.register_function(calculator, name="calculator")
    registry.register_function(boom, name="boom")
    return registry


def make_worker(name: str, skills: List[str], script: List[Any], tools: List[str] | None = None) -> Worker:
    registry = build_registry()
    if tools is not None:
        keep = ToolRegistry()
        for tool_name in tools:
            keep.register(registry.get(tool_name))
        registry = keep
    runtime = AgentRuntime(ScriptedBackend(script), registry, config=AgentConfig(max_steps=4), name=name)
    return Worker(name=name, runtime=runtime, skills=skills)


# --------------------------------------------------------------------------- #
# 任务拆分
# --------------------------------------------------------------------------- #

def test_default_decompose_splits_on_connectors():
    subtasks = default_decompose("查一下温度，然后检索规程，最后计算余量")
    assert [s.description for s in subtasks] == ["查一下温度", "检索规程", "计算余量"]
    assert [s.task_id for s in subtasks] == [1, 2, 3]


def test_default_decompose_dedups_and_drops_short_pieces():
    subtasks = default_decompose("查温度。查温度。好。再算一下余量")
    descriptions = [s.description for s in subtasks]
    assert descriptions == ["查温度", "再算一下余量"]


def test_default_decompose_respects_max_parts():
    subtasks = default_decompose("查温度，然后查振动，然后查电流，然后查压力，然后查转速", max_parts=2)
    assert len(subtasks) == 2


def test_default_decompose_handles_empty_and_single():
    assert default_decompose("") == []
    assert [s.description for s in default_decompose("只有一个任务")] == ["只有一个任务"]


# --------------------------------------------------------------------------- #
# Worker 选择
# --------------------------------------------------------------------------- #

def test_default_decompose_result_is_stable():
    first = [s.description for s in default_decompose("查温度，然后算余量")]
    second = [s.description for s in default_decompose("查温度，然后算余量")]
    assert first == second


def test_decompose_result_is_stable():
    first = [s.description for s in default_decompose("查温度，然后算余量")]
    second = [s.description for s in default_decompose("查温度，然后算余量")]
    assert first == second


def test_score_matches_skills_and_tool_names():
    collector = make_worker("采集", ["指标", "温度"], ["好"], tools=["query_metric"])
    researcher = make_worker("规程", ["规程"], ["好"], tools=["search_knowledge_base"])
    subtask = SubTask(1, "查一下 cnc-03 的温度指标")
    assert collector.score(subtask.description) > researcher.score(subtask.description)


def test_rank_is_stable_for_ties():
    a = make_worker("A", [], ["好"])
    b = make_worker("B", [], ["好"])
    supervisor = Supervisor([a, b])
    ranked = supervisor.rank(SubTask(1, "完全无关的任务描述"))
    assert [w.name for w in ranked] == ["A", "B"], "同分时必须保持注册顺序，结果才可复现"


def test_supervisor_requires_workers():
    with pytest.raises(ValueError):
        Supervisor([])


# --------------------------------------------------------------------------- #
# 端到端协作
# --------------------------------------------------------------------------- #

def test_run_routes_subtasks_to_best_worker_and_aggregates():
    collector = make_worker(
        "采集", ["指标", "温度"],
        [{"tool_calls": [{"name": "query_metric", "arguments": {"device": "cnc-03"}}]}, "温度 62.8。"],
        tools=["query_metric"],
    )
    researcher = make_worker(
        "规程", ["规程"],
        [{"tool_calls": [{"name": "search_knowledge_base", "arguments": {"query": "温升"}}]}, "按 sop 处置。"],
        tools=["search_knowledge_base"],
    )
    supervisor = Supervisor([collector, researcher], synthesized=False)

    result = supervisor.run("查一下温度指标，然后检索规程")
    assert result.success is True
    assert result.reason == "completed"
    assert len(result.subtasks) == 2
    assert [s.assignee for s in result.subtasks] == ["采集", "规程"]
    assert all(s.status == "done" for s in result.subtasks)
    assert "温度 62.8" in result.answer and "sop" in result.answer
    assert "[1]" in result.answer and "[2]" in result.answer


def test_failed_subtask_is_reassigned_to_next_worker():
    """最优 Worker 失败 → 改派给次优 Worker，而不是直接判失败。"""
    failing = make_worker(
        "会失败的采集", ["指标", "温度"],
        [{"tool_calls": [{"name": "boom", "arguments": {"x": 1}}]}, "不该到这里"],
        tools=["boom"],
    )
    backup = make_worker(
        "备用采集", ["指标", "温度"],
        [{"tool_calls": [{"name": "query_metric", "arguments": {"device": "cnc-03"}}]}, "备用通道拿到 62.8。"],
        tools=["query_metric"],
    )
    # 让 failing 的打分更高，确保它先被选中
    failing.skills = ("指标", "温度", "查询")
    supervisor = Supervisor([failing, backup], synthesized=False, max_attempts_per_task=2)

    result = supervisor.run("查温度指标")
    assert result.success is True
    sub = result.subtasks[0]
    assert sub.assignee == "备用采集"
    assert sub.attempts == 2
    assert "62.8" in sub.answer
    kinds = [m.kind for m in result.bus.history]
    assert kinds.count("delegate") == 2
    assert "error" in kinds and "result" in kinds


def test_all_workers_fail_marks_subtask_failed():
    failing = make_worker(
        "失败A", ["指标"],
        [{"tool_calls": [{"name": "boom", "arguments": {"x": 1}}]}, "x"],
        tools=["boom"],
    )
    failing2 = make_worker(
        "失败B", ["指标"],
        [{"tool_calls": [{"name": "boom", "arguments": {"x": 2}}]}, "x"],
        tools=["boom"],
    )
    supervisor = Supervisor([failing, failing2], synthesized=False, max_attempts_per_task=2)
    result = supervisor.run("查指标")
    assert result.success is False
    assert result.reason == "failed"
    assert result.failed and result.failed[0].error
    assert "未完成" in result.answer


def test_partial_success_is_reported_honestly():
    good = make_worker(
        "好", ["温度"],
        [{"tool_calls": [{"name": "query_metric", "arguments": {"device": "a"}}]}, "拿到了。"],
        tools=["query_metric"],
    )
    bad = make_worker(
        "坏", ["规程"],
        [{"tool_calls": [{"name": "boom", "arguments": {"x": 1}}]}, "x"],
        tools=["boom"],
    )
    supervisor = Supervisor([good, bad], synthesized=False, max_attempts_per_task=1)
    result = supervisor.run("查温度，然后检索规程")
    assert result.success is False
    assert result.reason == "partial"
    assert len(result.completed) == 1 and len(result.failed) == 1
    assert "完成 1/2" in result.answer
    assert "未完成" in result.answer


def test_worker_exception_is_captured():
    worker = make_worker("会崩的", ["任务"], ["好"])

    def explode(task: str, context: str = ""):
        raise RuntimeError("Worker 进程炸了")

    worker.run = explode  # type: ignore[assignment]
    supervisor = Supervisor([worker], synthesized=False, max_attempts_per_task=1)
    result = supervisor.run("随便什么任务")
    assert result.success is False
    assert "Worker 进程炸了" in result.subtasks[0].error
    assert any(m.kind == "error" for m in result.bus.history)


def test_synthesis_uses_backend_and_falls_back():
    collector = make_worker(
        "采集", ["温度"],
        [{"tool_calls": [{"name": "query_metric", "arguments": {"device": "a"}}]}, "62.8。"],
        tools=["query_metric"],
    )
    backend = ScriptedBackend(["综合结论：设备温度 62.8 ℃，状态正常。"])
    supervisor = Supervisor([collector], backend=backend, synthesized=True)
    result = supervisor.run("查温度")
    assert result.answer == "综合结论：设备温度 62.8 ℃，状态正常。"
    # 汇总提示词里必须带上子任务结论
    assert "62.8" in backend.calls[0]["messages"][1].content


def test_synthesis_failure_falls_back_to_aggregate():
    collector = make_worker(
        "采集", ["温度"],
        [{"tool_calls": [{"name": "query_metric", "arguments": {"device": "a"}}]}, "62.8。"],
        tools=["query_metric"],
    )
    backend = ScriptedBackend([RuntimeError("汇总模型挂了")])
    supervisor = Supervisor([collector], backend=backend, synthesized=True)
    result = supervisor.run("查温度")
    assert "完成 1/1" in result.answer


def test_delegation_trace_and_bus_stats():
    worker = make_worker(
        "采集", ["温度"],
        [{"tool_calls": [{"name": "query_metric", "arguments": {"device": "a"}}]}, "62.8。"],
        tools=["query_metric"],
    )
    result = Supervisor([worker], synthesized=False).run("查温度")
    assert result.tracer.phases(PHASE_DELEGATE)
    stats = result.bus.stats()
    assert stats["by_kind"]["delegate"] == 1
    assert stats["by_kind"]["result"] == 1
    assert result.rounds == 1
    assert result.to_dict()["subtasks"][0]["assignee"] == "采集"


def test_bus_subscribe_handler_exception_is_isolated():
    bus = MessageBus()
    seen: List[str] = []

    def broken(_: BusMessage) -> None:
        raise RuntimeError("订阅方炸了")

    def healthy(msg: BusMessage) -> None:
        seen.append(msg.kind)

    bus.subscribe("delegate", broken)
    bus.subscribe("delegate", healthy)
    bus.publish(BusMessage(sender="s", recipient="w", kind="delegate", payload={}))
    assert seen == ["delegate"]
    assert len(bus.of_kind("delegate")) == 1


def test_custom_decomposer_is_used():
    worker = make_worker("通用", [], ["什么都行。"])
    supervisor = Supervisor([worker], synthesized=False,
                            decomposer=lambda task: [SubTask(1, "自定义子任务")])
    result = supervisor.run("原始任务")
    assert result.subtasks[0].description == "自定义子任务"


def test_empty_decomposition_falls_back_to_whole_task():
    worker = make_worker("通用", [], ["完成。"])
    supervisor = Supervisor([worker], synthesized=False, decomposer=lambda task: [])
    result = supervisor.run("整体任务")
    assert result.subtasks[0].description == "整体任务"
    assert result.success is True
