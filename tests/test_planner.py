"""规划层测试：计划解析（含各种脏输出）、计划状态机、决策解析、提示词构造。"""

from __future__ import annotations

import pytest

from agentlab.planner import (
    Plan,
    PlanAndExecutePlanner,
    PlanParseError,
    PlanStep,
    ReActPlanner,
    load_json_block,
    parse_decision,
    parse_plan,
)


# --------------------------------------------------------------------------- #
# JSON 抽取
# --------------------------------------------------------------------------- #

def test_load_json_block_plain():
    assert load_json_block('{"a": 1}') == {"a": 1}


def test_load_json_block_with_fence_and_prose():
    text = '好的，这是计划：\n```json\n{"steps": ["a", "b"]}\n```\n希望对你有帮助。'
    assert load_json_block(text) == {"steps": ["a", "b"]}


def test_load_json_block_without_fence_but_with_prose():
    text = '分析如下 {"goal": "g", "steps": [{"description": "d"}]} 以上。'
    assert load_json_block(text)["goal"] == "g"


def test_load_json_block_raises_on_garbage():
    with pytest.raises(PlanParseError):
        load_json_block("完全不是 JSON")
    with pytest.raises(PlanParseError):
        load_json_block('{"a": 1,}')


# --------------------------------------------------------------------------- #
# 计划解析
# --------------------------------------------------------------------------- #

def test_parse_plan_full_form():
    text = (
        '{"goal": "排查温升", "steps": ['
        '{"description": "取温度序列", "tool": "query_metric"},'
        '{"description": "拟合趋势", "tool_hint": "trend"},'
        '{"description": "给出结论"}]}'
    )
    plan = parse_plan(text)
    assert plan.goal == "排查温升"
    assert len(plan) == 3
    assert [s.step_id for s in plan.steps] == [1, 2, 3]
    assert plan.steps[0].tool_hint == "query_metric"
    assert plan.steps[1].tool_hint == "trend"
    assert plan.steps[2].tool_hint is None
    assert all(s.status == "pending" for s in plan.steps)


def test_parse_plan_accepts_string_steps_and_alias_keys():
    plan = parse_plan('{"steps": ["第一步", {"task": "第二步", "tool_name": "calculator"}]}')
    assert [s.description for s in plan.steps] == ["第一步", "第二步"]
    assert plan.steps[1].tool_hint == "calculator"


def test_parse_plan_accepts_bare_list():
    plan = parse_plan('["a", "b"]', goal="G")
    assert plan.goal == "G" and len(plan) == 2


def test_parse_plan_falls_back_to_given_goal():
    plan = parse_plan('{"steps": ["a"]}', goal="外部目标")
    assert plan.goal == "外部目标"


@pytest.mark.parametrize(
    "text",
    [
        "没有 JSON",
        '{"steps": []}',
        '{"steps": [{"tool": "x"}]}',
        '{"steps": [123]}',
        '{"steps": "not-a-list"}',
    ],
)
def test_parse_plan_raises_on_invalid_shapes(text):
    with pytest.raises(PlanParseError):
        parse_plan(text)


# --------------------------------------------------------------------------- #
# 计划状态机
# --------------------------------------------------------------------------- #

def test_plan_progress_and_render():
    plan = Plan(goal="目标", steps=[PlanStep(1, "取数"), PlanStep(2, "分析"), PlanStep(3, "出结论")])
    assert plan.next_step().step_id == 1
    plan.mark(1, "done")
    assert plan.completion() == pytest.approx(1 / 3, abs=1e-4)
    assert plan.next_step().step_id == 2
    assert [s.step_id for s in plan.done] == [1]
    assert [s.step_id for s in plan.pending] == [2, 3]

    plan.mark(2, "failed", note="接口超时")
    rendered = plan.render()
    assert "[x] 1. 取数" in rendered
    assert "[!] 2. 分析" in rendered
    assert "接口超时" in rendered
    assert "[ ] 3. 出结论" in rendered
    assert "目标: 目标" in rendered


def test_plan_mark_unknown_step_is_noop():
    plan = Plan(goal="g", steps=[PlanStep(1, "a")])
    plan.mark(99, "done")
    assert plan.completion() == 0.0


def test_empty_plan():
    plan = Plan(goal="g")
    assert len(plan) == 0
    assert plan.next_step() is None
    assert plan.completion() == 0.0
    assert plan.render() == "（空计划）"


def test_plan_to_dict_is_serializable():
    import json

    plan = Plan(goal="g", steps=[PlanStep(1, "a", tool_hint="t")])
    payload = json.loads(json.dumps(plan.to_dict(), ensure_ascii=False))
    assert payload["steps"][0]["tool_hint"] == "t"
    assert payload["completion"] == 0.0


# --------------------------------------------------------------------------- #
# 决策解析
# --------------------------------------------------------------------------- #

def test_parse_decision_tool_action():
    decision = parse_decision('{"action": "tool", "tool": "query_metric", "arguments": {"device": "a"}}')
    assert decision["action"] == "tool"
    assert decision["tool"] == "query_metric"
    assert decision["arguments"] == {"device": "a"}


def test_parse_decision_accepts_aliases():
    assert parse_decision('{"action": "call", "tool": "t"}')["action"] == "tool"
    assert parse_decision('{"action": "tool_call", "tool": "t"}')["action"] == "tool"


def test_parse_decision_final_action():
    decision = parse_decision('{"action": "final", "answer": "完成了"}')
    assert decision == {"action": "final", "answer": "完成了", "thought": ""}


def test_parse_decision_degrades_to_final_on_plain_text():
    decision = parse_decision("我直接回答：温度正常。")
    assert decision["action"] == "final"
    assert "温度正常" in decision["answer"]


def test_parse_decision_tool_without_name_is_final():
    assert parse_decision('{"action": "tool"}')["action"] == "final"


# --------------------------------------------------------------------------- #
# 提示词构造
# --------------------------------------------------------------------------- #

def test_react_system_prompt_contains_catalog_and_constraints():
    planner = ReActPlanner(tool_catalog="- query_metric(device:string)", max_steps=5)
    prompt = planner.system_prompt()
    assert "query_metric(device:string)" in prompt
    assert "最多允许 5 步" in prompt
    assert "只做一件事" in prompt


def test_react_system_prompt_without_tools():
    assert "（无可用工具）" in ReActPlanner().system_prompt()


def test_react_step_prompt_lists_observations():
    planner = ReActPlanner()
    prompt = planner.step_prompt("查温度", ["温度 81.3", "振动 5.2"], hint="注意单位")
    assert "查温度" in prompt
    assert "1. 温度 81.3" in prompt and "2. 振动 5.2" in prompt
    assert "提示：注意单位" in prompt
    assert "目前还没有任何观察结果" in planner.step_prompt("查温度", [])


def test_plan_and_execute_prompt():
    planner = PlanAndExecutePlanner(tool_catalog="- t", max_steps=4)
    prompt = planner.plan_prompt("排查故障", context="资料")
    assert "只输出一个 JSON 对象" in prompt
    assert "2~4 步" in prompt
    assert "资料" in prompt


def test_replan_prompt_mentions_failure_and_done():
    planner = PlanAndExecutePlanner(tool_catalog="- t")
    plan = Plan(goal="g", steps=[PlanStep(1, "取数"), PlanStep(2, "分析")])
    plan.mark(1, "done")
    prompt = planner.replan_prompt("g", plan, "接口超时")
    assert "接口超时" in prompt
    assert "已完成：取数" in prompt
    assert "不要重复已经完成的步骤" in prompt


def test_merge_keeps_done_steps_and_renumbers():
    planner = PlanAndExecutePlanner()
    plan = Plan(goal="g", steps=[PlanStep(1, "step1"), PlanStep(2, "step2"), PlanStep(3, "step3")])
    plan.mark(1, "done")
    plan.mark(2, "failed", note="err")
    fresh = Plan(goal="g", steps=[PlanStep(1, "新 step2"), PlanStep(2, "新 step3")])
    merged = planner.merge(plan, fresh)
    assert [s.description for s in merged.steps] == ["step1", "新 step2", "新 step3"]
    assert [s.step_id for s in merged.steps] == [1, 2, 3]
    assert merged.revision == plan.revision + 1
    assert merged.done[0].description == "step1"
    assert all(s.status == "pending" for s in merged.pending)
