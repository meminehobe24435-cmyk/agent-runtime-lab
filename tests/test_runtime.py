"""运行时测试：主循环、四条自我修正路径、上下文压缩、计划与重规划、trace。"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agentlab.llm import Message, RuleBasedBackend, ScriptedBackend, ToolCall
from agentlab.memory import ShortTermMemory, WorkingMemory
from agentlab.rag import KnowledgeBase
from agentlab.runtime import AgentConfig, AgentRuntime
from agentlab.tools import ToolRegistry, ToolValidationError
from agentlab.trace import (
    PHASE_ACT,
    PHASE_CORRECT,
    PHASE_ERROR,
    PHASE_FINISH,
    PHASE_OBSERVE,
    PHASE_PLAN,
    PHASE_START,
)


# --------------------------------------------------------------------------- #
# 测试用工具
# --------------------------------------------------------------------------- #

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def query_metric(device: str, metric: str, window_min: int = 10) -> Dict[str, Any]:
        """查询设备指标。

        Args:
            device: 设备编号。
            metric: 指标名。
            window_min: 窗口分钟。
        """
        return {"device": device, "metric": metric, "window_min": window_min, "value": 62.8}

    def boom(x: int) -> int:
        """总是失败。

        Args:
            x: 参数。
        """
        raise RuntimeError("后端服务不可用")

    state = {"calls": 0}

    def flaky() -> str:
        """前两次失败。"""
        state["calls"] += 1
        if state["calls"] < 3:
            raise TimeoutError("网关超时")
        return "连通"

    registry.register_function(
        query_metric, name="query_metric",
        constraints={"window_min": {"minimum": 1, "maximum": 120}},
    )
    registry.register_function(boom, name="boom")
    registry.register_function(flaky, name="flaky", description="不稳定工具")
    return registry


def call(name: str, **arguments: Any) -> Dict[str, Any]:
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


def runtime_with(script: List[Any], config: AgentConfig | None = None, **kwargs: Any) -> AgentRuntime:
    return AgentRuntime(
        ScriptedBackend(script),
        build_registry(),
        config=config or AgentConfig(max_steps=6),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #

def test_single_tool_then_answer():
    agent = runtime_with([
        call("query_metric", device="cnc-03", metric="temperature"),
        "温度 62.8 ℃。",
    ])
    result = agent.run("查 cnc-03 温度")
    assert result.success is True
    assert result.reason == "completed"
    assert result.answer == "温度 62.8 ℃。"
    assert result.tool_calls == ["query_metric"]
    assert result.successful_tools == ["query_metric"]
    assert result.steps == 2
    phases = [e.phase for e in result.tracer.events]
    assert phases[0] == PHASE_START
    assert PHASE_ACT in phases and PHASE_OBSERVE in phases
    assert phases[-1] == PHASE_FINISH


def test_no_tool_needed_finishes_in_one_step():
    agent = runtime_with(["直接回答。"])
    result = agent.run("你好")
    assert result.success and result.steps == 1 and result.tool_calls == []


def test_multiple_tool_calls_in_one_message():
    agent = runtime_with([
        {"tool_calls": [
            {"name": "query_metric", "arguments": {"device": "a", "metric": "temperature"}},
            {"name": "query_metric", "arguments": {"device": "b", "metric": "vibration"}},
        ]},
        "两个都查到了。",
    ])
    result = agent.run("分别查两台设备")
    assert result.success
    assert result.tool_calls == ["query_metric", "query_metric"]
    assert result.steps == 2


def test_tool_arguments_are_the_ones_validated():
    agent = runtime_with([call("query_metric", device="cnc-03", metric="temperature"), "好"])
    result = agent.run("查")
    act = result.tracer.tool_calls()[0]
    assert act.data["arguments"] == {"device": "cnc-03", "metric": "temperature"}
    assert act.data["ok"] is True


# --------------------------------------------------------------------------- #
# 自我修正（四条路径）
# --------------------------------------------------------------------------- #

def test_invalid_arguments_are_blocked_before_execution():
    called: List[Any] = []

    def spy(x: int) -> int:
        """记录调用。

        Args:
            x: 参数。
        """
        called.append(x)
        return x

    registry = ToolRegistry()
    registry.register_function(spy, name="spy", constraints={"x": {"minimum": 1}})
    agent = AgentRuntime(
        ScriptedBackend([call("spy", x=0), call("spy", x=5), "done"]),
        registry,
        config=AgentConfig(max_steps=4),
    )
    result = agent.run("调用 spy")
    assert called == [5], "第一次非法参数不得执行；修正后才执行"
    assert result.corrections == 1
    kinds = [e.data.get("kind") for e in result.tracer.corrections()]
    assert "invalid_arguments" in kinds
    assert result.success is True


def test_invalid_argument_observation_contains_field_errors_and_signature():
    backend = ScriptedBackend([call("query_metric", device="a", metric="t", window_min=0), "结束"])
    agent = AgentRuntime(backend, build_registry(), config=AgentConfig(max_steps=4))
    result = agent.run("查")

    correct = result.tracer.corrections()[0]
    assert "小于最小值 1" in correct.detail

    # 回填给模型的 tool 消息里必须带字段级错误和正确签名
    tool_messages = [m for m in backend.calls[1]["messages"] if m.role == "tool"]
    assert tool_messages, "必须把错误回填给模型"
    content = tool_messages[-1].content
    assert "参数校验失败" in content
    assert "正确签名" in content
    assert "query_metric(" in content
    assert tool_messages[-1].tool_call_id == "call_1"


def test_unknown_tool_suggests_candidates():
    agent = runtime_with([call("query_metric_v2", device="a"),
                          call("query_metric", device="a", metric="t"), "好"])
    result = agent.run("查")
    assert result.corrections == 1
    assert result.tracer.corrections()[0].data["kind"] == "unknown_tool"
    assert result.successful_tools == ["query_metric"]


def test_tool_error_is_reported_and_recoverable():
    agent = runtime_with([call("flaky"), call("flaky"), call("flaky"), "已连通"])
    result = agent.run("探测网关")
    assert result.errors == 2
    assert result.corrections == 0, "工具内部报错不算参数纠正"
    assert result.success is True
    assert result.tool_calls == ["flaky", "flaky", "flaky"], "失败后的重试不得被死循环拦截"
    assert result.successful_tools == ["flaky"]
    failures = [e for e in result.tracer.tool_calls() if not e.data.get("ok", True)]
    assert len(failures) == 2


def test_loop_detection_blocks_third_identical_call():
    args = {"device": "p", "metric": "current"}
    agent = runtime_with([
        call("query_metric", **args),
        call("query_metric", **args),
        call("query_metric", **args),
        call("query_metric", **args),
        "停止重复查询。",
    ])
    result = agent.run("反复查")
    assert result.tool_calls == ["query_metric", "query_metric"], "第 3 次起必须被拦截"
    assert result.corrections == 2
    kinds = [e.data.get("kind") for e in result.tracer.corrections()]
    assert kinds.count("loop_detected") == 2
    assert result.success is True


def test_loop_detection_allows_different_arguments():
    agent = runtime_with([
        call("query_metric", device="a", metric="temperature"),
        call("query_metric", device="b", metric="temperature"),
        call("query_metric", device="c", metric="temperature"),
        "都查完了。",
    ])
    result = agent.run("查三台")
    assert len(result.tool_calls) == 3
    assert result.corrections == 0


# --------------------------------------------------------------------------- #
# 终止与异常
# --------------------------------------------------------------------------- #

def test_max_steps_exceeded_is_a_clean_failure():
    args = {"device": "p", "metric": "current"}
    script = [call("query_metric", **{"device": "d%d" % i, "metric": "current"}) for i in range(10)]
    agent = runtime_with(script, AgentConfig(max_steps=3))
    result = agent.run("永远查不完")
    assert result.success is False
    assert result.reason == "max_steps_exceeded"
    assert result.steps == 3
    assert result.tracer.phases(PHASE_FINISH)[-1].data["reason"] == "max_steps_exceeded"


def test_empty_answer_is_not_success():
    agent = runtime_with([""])
    result = agent.run("随便")
    assert result.success is False and result.reason == "empty_answer"


def test_backend_exception_is_captured():
    agent = runtime_with([RuntimeError("模型服务 500")])
    result = agent.run("随便")
    assert result.success is False
    assert result.reason == "backend_error"
    assert "模型服务 500" in result.answer
    assert result.errors == 1
    assert result.tracer.phases(PHASE_ERROR)


def test_stop_on_tool_error_aborts_immediately():
    agent = runtime_with([call("boom", x=1), "不该走到这里"], AgentConfig(stop_on_tool_error=True))
    result = agent.run("触发失败")
    assert result.success is False
    assert result.reason == "tool_error"
    assert result.errors == 1


def test_scripted_backend_exhaustion_is_visible():
    agent = runtime_with([call("query_metric", device="a", metric="temperature")])
    result = agent.run("查")
    assert "脚本已耗尽" in result.answer


def test_backend_receives_tool_schemas():
    backend = ScriptedBackend(["ok"])
    agent = AgentRuntime(backend, build_registry())
    agent.run("随便")
    names = {t["function"]["name"] for t in backend.calls[0]["tools"]}
    assert {"query_metric", "boom", "flaky"} <= names
    assert backend.calls[0]["messages"][0].role == "system"


# --------------------------------------------------------------------------- #
# RAG 与记忆
# --------------------------------------------------------------------------- #

def test_rag_context_is_injected_into_system_prompt():
    kb = KnowledgeBase(dim=512, size=200, overlap=40)
    kb.add_document("sop-overheat", "设备过热处置规程：温度升幅超过 15 摄氏度判定为温升异常。")
    backend = ScriptedBackend(["好"])
    agent = AgentRuntime(backend, build_registry(), knowledge=kb)
    result = agent.run("温度涨得太快怎么办")

    system = backend.calls[0]["messages"][0].content
    assert "【检索到的资料】" in system
    assert "sop-overheat" in system
    retrieve_events = [e for e in result.tracer.events if "检索到" in e.detail]
    assert retrieve_events and retrieve_events[0].data["hits"][0]["id"].startswith("sop-overheat")


def test_rag_can_be_disabled():
    kb = KnowledgeBase()
    kb.add_document("d", "内容")
    backend = ScriptedBackend(["好"])
    agent = AgentRuntime(backend, build_registry(), knowledge=kb,
                         config=AgentConfig(retrieve_before_plan=False))
    agent.run("问题")
    assert "【检索到的资料】" not in backend.calls[0]["messages"][0].content


def test_working_memory_records_successful_tool_results():
    agent = runtime_with([call("query_metric", device="a", metric="temperature"), "好"])
    result = agent.run("查")
    fact = result.working.provenance("last_query_metric")
    assert fact is not None
    assert fact.source == "tool:query_metric"
    assert fact.step == 1


def test_short_term_memory_tracks_conversation():
    memory = ShortTermMemory(max_tokens=1000)
    agent = runtime_with([call("query_metric", device="a", metric="temperature"), "温度 62.8。"],
                         memory=memory)
    agent.run("查温度")
    roles = [t.role for t in memory.turns]
    assert roles == ["user", "assistant"]
    assert memory.turns[0].content == "查温度"


# --------------------------------------------------------------------------- #
# 上下文压缩
# --------------------------------------------------------------------------- #

def test_context_compaction_fires_and_keeps_tool_pairing():
    script = []
    for i in range(6):
        script.append(call("query_metric", device=f"d{i}", metric="temperature"))
    script.append("完成。")
    backend = ScriptedBackend(script)
    agent = AgentRuntime(
        backend,
        build_registry(),
        config=AgentConfig(max_steps=8, context_budget_tokens=120),
    )
    result = agent.run("依次查六台设备并汇总")

    kinds = [e.data.get("kind") for e in result.tracer.corrections()]
    assert "context_compaction" in kinds or any(
        "压缩" in e.detail for e in result.tracer.events
    ), "应当触发过上下文压缩"

    messages = backend.calls[-1]["messages"]
    # 压缩后仍必须是合法的 function calling 消息序列
    assert messages[0].role == "system"
    assert messages[1].role != "tool", "压缩不得让 tool 消息成为首条非 system 消息"
    for idx, msg in enumerate(messages):
        if msg.role == "tool":
            assert any(m.tool_calls for m in messages[:idx]), "tool 消息必须有配对的 assistant.tool_calls"


def test_compaction_uses_summarizer_when_available():
    script = [call("query_metric", device=f"d{i}", metric="temperature") for i in range(6)] + ["完成。"]
    seen: List[str] = []
    agent = AgentRuntime(
        ScriptedBackend(script),
        build_registry(),
        config=AgentConfig(max_steps=8, context_budget_tokens=120),
        summarizer=lambda text: seen.append(text) or "（压缩摘要）",
    )
    agent.run("依次查六台设备")
    assert seen, "summarizer 应当被调用"
    assert "（压缩摘要）" in agent.backend.calls[-1]["messages"][0].content


def test_no_compaction_when_within_budget():
    agent = runtime_with([call("query_metric", device="a", metric="temperature"), "好"],
                         AgentConfig(context_budget_tokens=100000))
    result = agent.run("查")
    assert not any("压缩" in e.detail for e in result.tracer.events)


# --------------------------------------------------------------------------- #
# 计划与重规划
# --------------------------------------------------------------------------- #

PLAN_JSON = '{"goal": "排查温升", "steps": [{"description": "取温度", "tool": "query_metric"}, {"description": "给结论"}]}'


def test_plan_first_embeds_plan_into_system_prompt():
    backend = ScriptedBackend([PLAN_JSON, "结论：温升异常。"])
    agent = AgentRuntime(backend, build_registry(), config=AgentConfig(plan_first=True))
    result = agent.run("排查 cnc-03 温升")
    assert result.plan is not None and len(result.plan) == 2
    assert "当前计划" in backend.calls[1]["messages"][0].content
    assert any(e.phase == PHASE_PLAN for e in result.tracer.events)


def test_plan_parse_failure_degrades_to_react():
    backend = ScriptedBackend(["这不是 JSON", "直接回答。"])
    agent = AgentRuntime(backend, build_registry(), config=AgentConfig(plan_first=True))
    result = agent.run("随便")
    assert result.plan is None
    assert result.success is True
    kinds = [e.data.get("kind") for e in result.tracer.corrections()]
    assert "plan_parse_failed" in kinds


def test_replan_after_tool_failure_keeps_done_steps():
    backend = ScriptedBackend([
        PLAN_JSON,                       # 初次规划
        call("boom", x=1),               # 第 1 步失败
        '{"steps": ["改用 flaky"]}',     # 重规划
        call("flaky"),
        call("flaky"),
        call("flaky"),
        "改用备用通道后完成。",
    ])
    agent = AgentRuntime(backend, build_registry(), config=AgentConfig(plan_first=True, max_steps=8))
    result = agent.run("排查故障")
    assert result.replans == 1
    assert result.plan is not None and result.plan.revision == 1
    assert result.success is True
    assert any(e.data.get("kind") == "replan" for e in result.tracer.events)


def test_replan_failure_is_tolerated():
    backend = ScriptedBackend([PLAN_JSON, call("boom", x=1), "不是 JSON 的计划", "结束"])
    agent = AgentRuntime(backend, build_registry(), config=AgentConfig(plan_first=True))
    result = agent.run("排查故障")
    assert result.replans == 0
    assert any(e.data.get("kind") == "replan_failed" for e in result.tracer.events)


def test_explicit_plan_can_be_passed_in():
    from agentlab.planner import Plan, PlanStep

    plan = Plan(goal="g", steps=[PlanStep(1, "做点什么")])
    backend = ScriptedBackend(["做完了。"])
    agent = AgentRuntime(backend, build_registry())
    result = agent.run("任务", plan=plan)
    assert result.plan is plan
    assert "做点什么" in backend.calls[0]["messages"][0].content


# --------------------------------------------------------------------------- #
# 结果与 trace 序列化
# --------------------------------------------------------------------------- #

def test_result_to_dict_is_json_serializable():
    import json

    agent = runtime_with([call("query_metric", device="a", metric="temperature"), "好"])
    result = agent.run("查")
    payload = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert payload["success"] is True
    assert payload["successful_tools"] == ["query_metric"]
    assert payload["steps"] == 2


def test_tracer_can_be_disabled():
    agent = runtime_with([call("query_metric", device="a", metric="temperature"), "好"],
                         AgentConfig(trace_enabled=False))
    result = agent.run("查")
    assert result.tracer.events == []
    assert result.success is True
    assert result.tool_calls == ["query_metric"]


def test_config_to_dict_is_complete():
    config = AgentConfig()
    payload = config.to_dict()
    for key in ("max_steps", "max_tool_retries", "max_identical_calls", "max_replans",
                "context_budget_tokens", "plan_first", "trace_enabled"):
        assert key in payload


def test_rule_based_backend_drives_runtime():
    backend = RuleBasedBackend([
        {"match": ("温度",), "tool": "query_metric",
         "arguments": {"device": "a", "metric": "temperature"},
         "until": "observed", "answer": "温度正常。"},
    ])
    agent = AgentRuntime(backend, build_registry())
    result = agent.run("看看温度")
    assert result.success is True
    assert result.tool_calls == ["query_metric"]
    assert result.answer == "温度正常。"


def test_rule_based_backend_without_match_answers_directly():
    backend = RuleBasedBackend([{"match": ("温度",), "tool": "query_metric", "answer": "x"}])
    agent = AgentRuntime(backend, build_registry())
    result = agent.run("无关问题")
    assert result.success is True
    assert "没有匹配到" in result.answer


def test_rule_based_backend_reports_missing_tool():
    backend = RuleBasedBackend([{"match": ("温度",), "tool": "not_registered", "answer": "x"}])
    agent = AgentRuntime(backend, build_registry())
    result = agent.run("温度多少")
    assert result.success is True
    assert "缺少完成该任务所需的工具" in result.answer
