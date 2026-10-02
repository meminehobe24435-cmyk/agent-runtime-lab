"""可观测性测试：事件、阶段、统计、导出。"""

from __future__ import annotations

import json
import time

from agentlab.trace import (
    PHASE_ACT,
    PHASE_CORRECT,
    PHASE_FINISH,
    PHASE_OBSERVE,
    PHASE_START,
    TraceEvent,
    Tracer,
)


def test_step_counter_and_events():
    tracer = Tracer(name="a")
    tracer.event(PHASE_START, "开始")
    tracer.step_start()
    tracer.event(PHASE_ACT, "调用工具", {"tool": "t", "ok": True})
    tracer.event(PHASE_OBSERVE, "看结果", {"tool": "t"})
    tracer.step_start()
    tracer.event(PHASE_FINISH, "结束")

    assert tracer.step == 2
    assert len(tracer.events) == 4
    assert [e.step for e in tracer.events] == [0, 1, 1, 2]
    assert tracer.tool_calls() == [tracer.events[1]]
    assert tracer.phases(PHASE_OBSERVE) == [tracer.events[2]]
    assert all(e.agent == "a" for e in tracer.events)


def test_summary_counts_tools_failures_and_corrections():
    tracer = Tracer(name="x")
    tracer.step_start()
    tracer.event(PHASE_ACT, "ok", {"tool": "t1", "ok": True})
    tracer.event(PHASE_ACT, "fail", {"tool": "t2", "ok": False})
    tracer.event(PHASE_CORRECT, "纠正", {"tool": "t2"})
    tracer.event(PHASE_ACT, "ok2", {"tool": "t1", "ok": True})
    summary = tracer.summary()
    assert summary["steps"] == 1
    assert summary["tool_calls"] == 3
    assert summary["tool_names"] == ["t1", "t2", "t1"]
    assert summary["tool_failures"] == 1
    assert summary["corrections"] == 1
    assert summary["errors"] == 1
    assert summary["duration_ms"] >= 0


def test_default_ok_is_true_when_not_specified():
    tracer = Tracer()
    tracer.step_start()
    tracer.event(PHASE_ACT, "没有 ok 字段", {"tool": "t"})
    assert tracer.summary()["tool_failures"] == 0


def test_disabled_tracer_records_nothing_but_still_returns_events():
    tracer = Tracer(name="quiet", enabled=False)
    event = tracer.event(PHASE_START, "不该被记录")
    assert tracer.events == []
    assert isinstance(event, TraceEvent) and event.detail == "不该被记录"
    assert tracer.summary()["events"] == 0


def test_elapsed_ms_is_measured_per_event():
    tracer = Tracer()
    tracer.step_start()
    time.sleep(0.01)
    event = tracer.event(PHASE_ACT, "慢动作")
    assert event.elapsed_ms >= 5.0


def test_event_to_dict_omits_empty_fields():
    tracer = Tracer(name="a")
    tracer.event(PHASE_START, "无数据")
    payload = tracer.events[0].to_dict()
    assert "data" not in payload
    assert payload["agent"] == "a"

    tracer.event(PHASE_ACT, "有数据", {"tool": "t"}, agent="b")
    payload = tracer.events[1].to_dict()
    assert payload["data"] == {"tool": "t"} and payload["agent"] == "b"


def test_to_json_and_save(tmp_path):
    tracer = Tracer(name="a")
    tracer.step_start()
    tracer.event(PHASE_ACT, "调用", {"tool": "t", "ok": True, "arguments": {"x": 1}})

    data = json.loads(tracer.to_json())
    assert data["name"] == "a"
    assert data["summary"]["tool_calls"] == 1
    assert data["events"][0]["data"]["arguments"] == {"x": 1}

    path = tmp_path / "trace.json"
    tracer.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["summary"]["tool_calls"] == 1


def test_render_is_readable():
    tracer = Tracer(name="render-me")
    tracer.step_start()
    tracer.event(PHASE_ACT, "query_metric(...)", {"tool": "query_metric", "ok": True})
    tracer.event(PHASE_ACT, "boom(...)", {"tool": "boom", "ok": False})
    tracer.event(PHASE_FINISH, "结束")

    text = tracer.render()
    assert "# trace: render-me" in text
    assert "[query_metric] ok" in text
    assert "[boom] FAILED" in text
    assert "steps=1 tools=2 failures=1 corrections=0" in text


def test_render_on_empty_tracer():
    text = Tracer(name="empty").render()
    assert "# trace: empty" in text
    assert "steps=0 tools=0" in text
