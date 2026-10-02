"""可观测性：结构化 trace。

对应 JD 职责里的「提升 Agent 的可观测性」。

一次 Agent 运行会产出三类信息：**做了什么**（step / phase）、**花了多少**（耗时、token）、
**错在哪**（工具错误、参数校验失败、自我修正次数）。
本模块把它们统一成事件流，并能导出 JSON 或渲染成可读文本 ——
评测报告里的「自我修正恢复率」等指标就是直接从 trace 里算出来的，
而不是靠运行时自己记账（少一份状态就少一处不一致）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 事件阶段
PHASE_START = "start"
PHASE_PLAN = "plan"
PHASE_ACT = "act"
PHASE_OBSERVE = "observe"
PHASE_CORRECT = "correct"
PHASE_FINISH = "finish"
PHASE_ERROR = "error"
PHASE_DELEGATE = "delegate"

PHASES = (
    PHASE_START, PHASE_PLAN, PHASE_ACT, PHASE_OBSERVE,
    PHASE_CORRECT, PHASE_FINISH, PHASE_ERROR, PHASE_DELEGATE,
)


@dataclass
class TraceEvent:
    step: int
    phase: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    agent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "step": self.step,
            "phase": self.phase,
            "detail": self.detail,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }
        if self.agent:
            out["agent"] = self.agent
        if self.data:
            out["data"] = self.data
        return out


class Tracer:
    """事件收集器。线程内使用即可（多智能体场景按 agent 名分桶）。"""

    def __init__(self, name: str = "agent", enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self.events: List[TraceEvent] = []
        self._started = time.perf_counter()
        self._step = 0
        self._mark = self._started

    # -- 记录 ------------------------------------------------------------- #
    def step_start(self) -> int:
        self._step += 1
        self._mark = time.perf_counter()
        return self._step

    @property
    def step(self) -> int:
        return self._step

    def _since_mark(self) -> float:
        now = time.perf_counter()
        delta = (now - self._mark) * 1000.0
        self._mark = now
        return delta

    def event(
        self,
        phase: str,
        detail: str = "",
        data: Optional[Dict[str, Any]] = None,
        agent: str = "",
    ) -> TraceEvent:
        ev = TraceEvent(
            step=self._step,
            phase=phase,
            detail=detail,
            data=dict(data or {}),
            elapsed_ms=self._since_mark(),
            agent=agent or self.name,
        )
        if self.enabled:
            self.events.append(ev)
        return ev

    # -- 统计 ------------------------------------------------------------- #
    @property
    def duration_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000.0

    def phases(self, phase: str) -> List[TraceEvent]:
        return [e for e in self.events if e.phase == phase]

    def tool_calls(self) -> List[TraceEvent]:
        return self.phases(PHASE_ACT)

    def corrections(self) -> List[TraceEvent]:
        return self.phases(PHASE_CORRECT)

    def errors(self) -> List[TraceEvent]:
        return [e for e in self.events if e.phase in (PHASE_ERROR, PHASE_CORRECT)]

    def summary(self) -> Dict[str, Any]:
        tools = [e.data.get("tool") for e in self.tool_calls()]
        failed = [e for e in self.tool_calls() if not e.data.get("ok", True)]
        return {
            "agent": self.name,
            "events": len(self.events),
            "steps": self._step,
            "tool_calls": len(tools),
            "tool_names": tools,
            "tool_failures": len(failed),
            "corrections": len(self.corrections()),
            "errors": len(self.errors()),
            "duration_ms": round(self.duration_ms, 3),
        }

    # -- 导出 ------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary(),
            "events": [e.to_dict() for e in self.events],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: Any) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def render(self) -> str:
        width = max((len(e.phase) for e in self.events), default=5)
        lines = [f"# trace: {self.name}"]
        for ev in self.events:
            tool = ev.data.get("tool")
            tag = f" [{tool}]" if tool else ""
            flag = ""
            if ev.phase == PHASE_ACT:
                flag = " ok" if ev.data.get("ok", True) else " FAILED"
            lines.append(
                f"{ev.step:>3} {ev.phase:<{width}} {ev.elapsed_ms:>8.2f}ms{tag}{flag} {ev.detail}"[:200]
            )
        s = self.summary()
        lines.append(
            f"--- steps={s['steps']} tools={s['tool_calls']} failures={s['tool_failures']} "
            f"corrections={s['corrections']} duration={s['duration_ms']:.1f}ms"
        )
        return "\n".join(lines)
