"""评测体系测试：判据、指标口径、报告导出、任务隔离、内置评测集。"""

from __future__ import annotations

import json
from typing import Any, List

import pytest

from agentlab.eval import (
    EvalReport,
    EvalTask,
    SuiteRunner,
    TaskOutcome,
    build_knowledge_base,
    build_ops_tools,
    build_suite_tasks,
    make_scenario_factory,
    run_default_suite,
    scenario_summary,
)
from agentlab.llm import ScriptedBackend
from agentlab.planner import Plan, PlanStep
from agentlab.runtime import AgentConfig, AgentResult, AgentRuntime
from agentlab.tools import ToolRegistry
from agentlab.trace import Tracer


# --------------------------------------------------------------------------- #
# 判据
# --------------------------------------------------------------------------- #

def make_result(**kwargs: Any) -> AgentResult:
    defaults = dict(
        task="t", answer="温度 62.8 ℃，判定为异常。", success=True, reason="completed",
        steps=2, tool_calls=["query_metric"], successful_tools=["query_metric"],
        corrections=0, replans=0, errors=0, duration_ms=1.0, tracer=Tracer(),
    )
    defaults.update(kwargs)
    return AgentResult(**defaults)


def test_check_passes_on_matching_result():
    task = EvalTask("t", "p", expected_tools=("query_metric",),
                    expected_answer_contains=("62.8", "异常"), max_steps=3)
    checks = task.check(make_result())
    assert all(checks.values()), checks


def test_check_detects_missing_answer_keyword():
    task = EvalTask("t", "p", expected_answer_contains=("不存在的内容",))
    checks = task.check(make_result())
    assert checks["答案要点"] is False
    assert TaskOutcome(task, make_result(), checks).failed_checks == ["答案要点"]


def test_check_uses_successful_tools_for_forbidden():
    """被拦下的工具名（未成功执行）不该算「用了禁用工具」。"""
    task = EvalTask("t", "p", forbidden_tools=("query_metric_v2",))
    result = make_result(tool_calls=["query_metric_v2", "query_metric"],
                         successful_tools=["query_metric"])
    assert task.check(result)["未用禁用工具"] is True

    bad = make_result(tool_calls=["query_metric_v2"], successful_tools=["query_metric_v2"])
    assert task.check(bad)["未用禁用工具"] is False


def test_check_sequence_and_step_limit():
    task = EvalTask("t", "p", expected_sequence=("a", "b"), max_steps=1)
    checks = task.check(make_result(tool_calls=["a", "b"], successful_tools=["a", "b"], steps=2))
    assert checks["工具序列严格匹配"] is True
    assert checks["未超步数"] is False


def test_check_replan_and_compaction():
    task = EvalTask("t", "p", expect_replans=1, expect_compaction=True)
    checks = task.check(make_result(replans=0))
    assert checks["发生重规划"] is False and checks["触发上下文压缩"] is False

    tracer = Tracer()
    tracer.event("correct", "上下文压缩：折叠 3 条历史消息", {"kind": "context_compaction"})
    checks = task.check(make_result(replans=2, tracer=tracer))
    assert checks["发生重规划"] is True and checks["触发上下文压缩"] is True


def test_outcome_without_result_never_passes():
    outcome = TaskOutcome(EvalTask("t", "p"), None, {}, error="boom")
    assert outcome.passed is False
    assert outcome.tool_calls == [] and outcome.successful_tools == []
    assert outcome.to_dict()["error"] == "boom"


# --------------------------------------------------------------------------- #
# 指标口径
# --------------------------------------------------------------------------- #

def build_report() -> EvalReport:
    clean = EvalTask("clean", "p", expected_tools=("t1",))
    recovered = EvalTask("recovered", "p", expected_tools=("t2",))
    failed = EvalTask("failed", "p", expected_tools=("t3",))
    outcomes = [
        TaskOutcome(clean, make_result(tool_calls=["t1"], successful_tools=["t1"]),
                    clean.check(make_result(tool_calls=["t1"], successful_tools=["t1"]))),
        TaskOutcome(recovered, make_result(errors=2, corrections=1,
                                           tool_calls=["t2", "t2"], successful_tools=["t2"]),
                    recovered.check(make_result(tool_calls=["t2"], successful_tools=["t2"]))),
        TaskOutcome(failed, make_result(success=False, answer="", errors=1,
                                        tool_calls=["t9"], successful_tools=[]),
                    {"收敛": False, "答案要点": False}),
    ]
    return EvalReport(outcomes=outcomes, label="unit")


def test_metrics_success_and_recovery_rates():
    metrics = build_report().metrics()
    assert metrics["任务数"] == 3
    assert metrics["通过"] == 2
    assert metrics["任务成功率"] == pytest.approx(2 / 3, abs=1e-4)
    # 出错任务 2 个（recovered、failed），其中 recovered 仍然通过
    assert metrics["出错任务数"] == 2
    assert metrics["自我修正恢复率"] == 0.5
    assert metrics["发生自我修正的任务"] == 1


def test_metrics_tool_precision_recall_and_redundancy():
    metrics = build_report().metrics()
    # clean 1/1、recovered 1/1、failed 0/1
    assert metrics["工具召回率"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["工具调用总数"] == 4  # t1, t2, t2, t9
    assert metrics["冗余调用"] == 1


def test_metrics_handles_empty_report():
    metrics = EvalReport().metrics()
    assert metrics["任务数"] == 0
    assert metrics["任务成功率"] == 0.0
    assert metrics["自我修正恢复率"] == 0.0


def test_report_exports_json_and_markdown():
    report = build_report()
    payload = json.loads(report.to_json())
    assert payload["label"] == "unit"
    assert len(payload["outcomes"]) == 3
    assert payload["metrics"]["任务数"] == 3

    markdown = report.to_markdown()
    assert "| 任务成功率 |" in markdown
    assert "| clean | ✅ |" in markdown
    assert "| failed | ❌ |" in markdown
    assert "脚本化模型后端" in markdown


def test_report_save(tmp_path):
    path = tmp_path / "report.json"
    build_report().save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["metrics"]["通过"] == 2


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #

def test_suite_runner_captures_factory_errors():
    def factory(task: EvalTask) -> AgentRuntime:
        if task.id == "boom":
            raise RuntimeError("工厂炸了")
        return AgentRuntime(ScriptedBackend(["完成"]), ToolRegistry())

    runner = SuiteRunner(factory, label="x")
    report = runner.run([EvalTask("ok", "p"), EvalTask("boom", "p")])
    assert report.total == 2
    assert report.passed == 1
    assert report.outcomes[1].error == "RuntimeError: 工厂炸了"
    assert report.outcomes[1].passed is False


def test_suite_runner_is_deterministic(tmp_path):
    """同一套任务跑两遍必须得到完全一样的指标（含工具调用序列）。

    耗时类指标天然会抖动，比较时排除掉 —— 但「顺序相关」的差异必须暴露出来。
    """
    tasks = build_suite_tasks()
    first = SuiteRunner(make_scenario_factory(tmp_path), label="a").run(tasks)
    second = SuiteRunner(make_scenario_factory(tmp_path), label="a").run(tasks)

    def stable(metrics: dict) -> dict:
        return {k: v for k, v in metrics.items() if not k.endswith("ms")}

    assert stable(first.metrics()) == stable(second.metrics())
    assert [o.tool_calls for o in first.outcomes] == [o.tool_calls for o in second.outcomes]
    assert [o.passed for o in first.outcomes] == [o.passed for o in second.outcomes]


def test_suite_tasks_are_isolated_from_each_other(tmp_path):
    """用例之间不能互相污染：单独跑与整轮跑的结果必须一致。

    这条测试是针对一个真实缺陷加的：带内部状态的工具（第一次调用必失败）
    曾被整轮评测共享，先跑的用例把「第一次失败」用掉之后，
    依赖该失败的用例就静默走了另一条路径，指标随顺序变化。
    """
    tasks = {t.id: t for t in build_suite_tasks()}
    target = tasks["plan-and-replan"]

    standalone = SuiteRunner(make_scenario_factory(tmp_path)).run([target]).outcomes[0]
    in_suite = {o.task.id: o for o in
                SuiteRunner(make_scenario_factory(tmp_path)).run(build_suite_tasks()).outcomes}[
        "plan-and-replan"]

    assert standalone.passed and in_suite.passed
    assert standalone.result is not None and in_suite.result is not None
    assert standalone.result.replans == in_suite.result.replans == 1
    assert standalone.tool_calls == in_suite.tool_calls


# --------------------------------------------------------------------------- #
# 内置场景
# --------------------------------------------------------------------------- #

def test_build_knowledge_base_content():
    kb = build_knowledge_base()
    assert len(kb.docs) == 3
    hits = kb.retrieve("温升异常怎么处置", top_k=1)
    assert hits and hits[0].doc_id == "sop-overheat"


def test_ops_tools_registry_shape(tmp_path):
    kb = build_knowledge_base()
    registry = build_ops_tools(kb, tmp_path)
    names = set(registry.names)
    assert {"query_metric", "trend", "convert_unit", "calculator",
            "search_knowledge_base", "write_report", "read_file",
            "write_file", "list_files", "search_files", "flaky_gateway"} <= names
    spec = registry.get("query_metric")
    assert spec.parameters["properties"]["device"]["enum"] == ["cnc-03", "press-01", "pump-07"]
    assert spec.parameters["properties"]["window_min"]["minimum"] == 1


def test_ops_tools_reject_bad_values(tmp_path):
    kb = build_knowledge_base()
    registry = build_ops_tools(kb, tmp_path)
    from agentlab.tools import ToolValidationError

    with pytest.raises(ToolValidationError):
        registry.call("query_metric", {"device": "no-such-device", "metric": "temperature"})
    with pytest.raises(ToolValidationError):
        registry.call("query_metric", {"device": "press-01", "metric": "temperature", "window_min": 0})

    result = registry.call("query_metric", {"device": "press-01", "metric": "temperature"})
    assert result.ok and result.value["series"] == [61.2, 63.8, 68.4, 74.9, 81.3]


def test_trend_tool_computes_slope(tmp_path):
    registry = build_ops_tools(build_knowledge_base(), tmp_path)
    result = registry.call("trend", {"values": [35.5, 40.2, 47.7, 55.1, 62.8]})
    assert result.ok
    # 最小二乘：Σ(x-x̄)(y-ȳ)/Σ(x-x̄)² = 69.5/10
    assert result.value["slope"] == pytest.approx(6.95, abs=0.001)
    assert result.value["r2"] > 0.99


def test_convert_unit_tool(tmp_path):
    from agentlab.tools import ToolValidationError

    registry = build_ops_tools(build_knowledge_base(), tmp_path)
    assert registry.call("convert_unit", {"value": 81.3, "from_unit": "celsius",
                                          "to_unit": "fahrenheit"}).value.startswith("178.34")
    assert registry.call("convert_unit", {"value": 1000, "from_unit": "mm/s",
                                          "to_unit": "m/s"}).value.startswith("1")
    # 不在枚举内的单位被前置拦截（而不是让工具跑起来再报错）
    with pytest.raises(ToolValidationError):
        registry.call("convert_unit", {"value": 1, "from_unit": "kg", "to_unit": "m/s"})

    # 同为合法单位但量纲不同 → 由工具自己报错，运行时把它变成观察结果
    weird = registry.call("convert_unit", {"value": 1, "from_unit": "celsius", "to_unit": "m/s"})
    assert weird.ok is False and "不支持" in weird.error


def test_trend_tool_rejects_single_point(tmp_path):
    from agentlab.tools import ToolValidationError

    registry = build_ops_tools(build_knowledge_base(), tmp_path)
    with pytest.raises(ToolValidationError):
        registry.call("trend", {"values": [1.0]})


def test_flaky_gateway_fails_once(tmp_path):
    registry = build_ops_tools(build_knowledge_base(), tmp_path)
    first = registry.call("flaky_gateway", {"device": "pump-07"})
    second = registry.call("flaky_gateway", {"device": "pump-07"})
    assert first.ok is False and "超时" in first.error
    assert second.ok is True and "心跳正常" in second.value


def test_write_report_stays_in_sandbox(tmp_path):
    registry = build_ops_tools(build_knowledge_base(), tmp_path)
    ok = registry.call("write_report", {"path": "reports/a.md", "content": "内容"})
    assert ok.ok and (tmp_path / "reports" / "a.md").read_text(encoding="utf-8") == "内容"
    escaped = registry.call("write_report", {"path": "../escape.md", "content": "x"})
    assert escaped.ok is False and "越界" in escaped.error


def test_scenario_summary(tmp_path):
    kb = build_knowledge_base()
    registry = build_ops_tools(kb, tmp_path)
    summary = scenario_summary(registry, kb)
    assert summary["tool_count"] == len(registry)
    assert summary["knowledge_docs"] == 3
    assert summary["tasks"] == len(build_suite_tasks())


def test_default_suite_all_pass(tmp_path):
    """内置评测集必须全绿 —— 这是「运行时改坏了」的总闸门。"""
    report = run_default_suite(tmp_path)
    failures = [(o.task.id, o.failed_checks, o.error) for o in report.outcomes if not o.passed]
    assert not failures, failures
    metrics = report.metrics()
    assert metrics["任务成功率"] == 1.0
    assert metrics["工具召回率"] == 1.0
    assert metrics["工具准确率"] == 1.0
    assert metrics["自我修正恢复率"] == 1.0
    # 四条自我修正路径都要被真实触发过，否则指标是"假绿"
    assert metrics["参数校验纠正次数"] >= 1
    assert metrics["未知工具纠正次数"] >= 1
    assert metrics["死循环拦截次数"] >= 1
    assert metrics["发生重规划的任务"] >= 1


def test_suite_includes_expected_coverage(tmp_path):
    tasks = build_suite_tasks()
    assert len(tasks) >= 12
    assert any(t.expect_compaction for t in tasks)
    assert any(t.plan_first for t in tasks)
    assert any(t.forbidden_tools for t in tasks)
    assert any(not t.expected_tools for t in tasks)
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "任务 id 必须唯一"
