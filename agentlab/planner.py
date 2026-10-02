"""任务规划与决策解析。

JD 职责 1 要的是「支持任务规划 …… 与自我修正」。运行时里规划分两种粒度：

- **ReAct（单步推理）**：``思考 → 调用工具 → 观察`` 循环，靠 :meth:`ReActPlanner.system_prompt`
  约束模型「一次只做一件事，拿到观察再决定下一步」；
- **Plan-and-Execute（先规划后执行）**：:class:`PlanAndExecutePlanner` 先让模型输出
  一份 JSON 计划，运行时逐步执行，某一步失败时允许**重新规划**（只保留已完成的步骤）。

本模块刻意做成**纯函数 + 数据类**：提示词怎么拼、模型输出的 JSON 怎么解析、
解析失败怎么降级，都能单独测；运行时不掺进这些细节，出错时也容易定位是哪一层的问题。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_PLAN_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.S)
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class PlanParseError(ValueError):
    """模型输出的计划无法解析。"""


@dataclass
class PlanStep:
    """计划中的一步。"""

    step_id: int
    description: str
    tool_hint: Optional[str] = None
    status: str = "pending"  # pending / running / done / failed / skipped
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_id,
            "description": self.description,
            "tool_hint": self.tool_hint,
            "status": self.status,
            "note": self.note,
        }


@dataclass
class Plan:
    """一份可执行计划。"""

    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    revision: int = 0

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def pending(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == "pending"]

    @property
    def done(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == "done"]

    def next_step(self) -> Optional[PlanStep]:
        for step in self.steps:
            if step.status == "pending":
                return step
        return None

    def mark(self, step_id: int, status: str, note: str = "") -> None:
        for step in self.steps:
            if step.step_id == step_id:
                step.status = status
                if note:
                    step.note = note
                return

    def completion(self) -> float:
        if not self.steps:
            return 0.0
        return round(len(self.done) / len(self.steps), 4)

    def render(self) -> str:
        if not self.steps:
            return "（空计划）"
        mark = {"pending": "[ ]", "running": "[~]", "done": "[x]", "failed": "[!]", "skipped": "[-]"}
        lines = [f"目标: {self.goal}"]
        for step in self.steps:
            hint = f" (建议工具: {step.tool_hint})" if step.tool_hint else ""
            note = f"  // {step.note}" if step.note else ""
            lines.append(f"{mark.get(step.status, '[?]')} {step.step_id}. {step.description}{hint}{note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "revision": self.revision,
            "completion": self.completion(),
            "steps": [s.to_dict() for s in self.steps],
        }


def _strip_fence(text: str) -> str:
    match = _PLAN_FENCE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def load_json_block(text: str) -> Any:
    """从模型输出里抽出第一个 JSON 对象。

    真实模型经常在 JSON 前后带解释文字或 markdown 围栏，这里做三级降级：
    去围栏 → 直接解析 → 正则截取最外层 ``{...}``。
    """
    raw = _strip_fence(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT.search(raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"JSON 解析失败: {exc.msg}") from None
    raise PlanParseError("输出中找不到 JSON 对象")


def parse_plan(text: str, goal: str = "") -> Plan:
    """把模型输出解析成 :class:`Plan`。

    兼容两种常见形状::

        {"goal": "...", "steps": [{"description": "...", "tool": "..."}]}
        {"steps": ["第一步", "第二步"]}
    """
    data = load_json_block(text)
    if isinstance(data, list):
        data = {"steps": data}
    if not isinstance(data, dict):
        raise PlanParseError(f"计划必须是对象或数组，收到 {type(data).__name__}")

    raw_steps = data.get("steps") or data.get("plan") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanParseError("计划里没有 steps 字段或 steps 为空")

    steps: List[PlanStep] = []
    for idx, raw in enumerate(raw_steps, start=1):
        if isinstance(raw, str):
            steps.append(PlanStep(step_id=idx, description=raw.strip()))
            continue
        if not isinstance(raw, dict):
            raise PlanParseError(f"第 {idx} 步格式非法: {type(raw).__name__}")
        desc = str(
            raw.get("description") or raw.get("step") or raw.get("task") or raw.get("action") or ""
        ).strip()
        if not desc:
            raise PlanParseError(f"第 {idx} 步缺少 description")
        hint = raw.get("tool") or raw.get("tool_hint") or raw.get("tool_name")
        steps.append(
            PlanStep(
                step_id=idx,
                description=desc,
                tool_hint=str(hint) if hint else None,
            )
        )
    return Plan(goal=str(data.get("goal") or goal), steps=steps)


def parse_decision(text: str) -> Dict[str, Any]:
    """解析单步决策 JSON（``{"action": "tool"|"final", ...}``）。解析失败时降级成最终回答。"""
    try:
        data = load_json_block(text)
    except PlanParseError:
        return {"action": "final", "answer": (text or "").strip()}
    if not isinstance(data, dict):
        return {"action": "final", "answer": str(data)}
    action = str(data.get("action") or "").strip().lower()
    if action in ("tool", "call", "tool_call") and data.get("tool"):
        return {
            "action": "tool",
            "tool": str(data["tool"]),
            "arguments": data.get("arguments") or {},
            "thought": str(data.get("thought") or ""),
        }
    return {
        "action": "final",
        "answer": str(data.get("answer") or data.get("final") or data.get("content") or "").strip(),
        "thought": str(data.get("thought") or ""),
    }


class ReActPlanner:
    """单步推理规划器：负责把「当前处境」拼成提示词。"""

    def __init__(self, tool_catalog: str = "", max_steps: int = 8) -> None:
        self.tool_catalog = tool_catalog
        self.max_steps = max_steps

    def system_prompt(self) -> str:
        parts = [
            "你是一个可调用工具的智能体。可用工具如下：",
            self.tool_catalog or "（无可用工具）",
            "",
            "工作方式：每一轮只做一件事 —— 要么调用一个工具获取信息，要么给出最终回答。",
            "拿到工具观察结果后再决定下一步，不要臆测工具返回的内容。",
            "如果某个工具报错，请阅读错误信息，修正参数后重试，或改用其他工具。",
            "信息足够时立刻给出最终回答，不要为了凑步骤重复调用同一个工具。",
            f"最多允许 {self.max_steps} 步。",
        ]
        return "\n".join(parts)

    def step_prompt(self, task: str, observations: Sequence[str], hint: str = "") -> str:
        lines = [f"任务：{task}"]
        if observations:
            lines.append("已有的观察结果：")
            lines.extend(f"{i}. {o}" for i, o in enumerate(observations, start=1))
        else:
            lines.append("目前还没有任何观察结果。")
        if hint:
            lines.append(f"提示：{hint}")
        lines.append("现在请决定下一步。")
        return "\n".join(lines)


class PlanAndExecutePlanner:
    """先规划后执行：拼规划提示词 + 执行期重规划提示词。"""

    def __init__(self, tool_catalog: str = "", max_steps: int = 6) -> None:
        self.tool_catalog = tool_catalog
        self.max_steps = max_steps

    def plan_prompt(self, task: str, context: str = "") -> str:
        lines = [
            "请把下面的任务拆解成可执行的步骤，并只输出一个 JSON 对象。",
            "格式：{\"goal\": \"...\", \"steps\": [{\"description\": \"...\", \"tool\": \"工具名或 null\"}]}",
            f"步骤数量控制在 2~{self.max_steps} 步，每步都要能独立验证。",
            "",
            f"任务：{task}",
        ]
        if context:
            lines.append(f"可用资料：\n{context}")
        lines.append(f"可用工具：\n{self.tool_catalog or '（无）'}")
        return "\n".join(lines)

    def replan_prompt(self, task: str, plan: Plan, failure: str) -> str:
        remaining = [s for s in plan.steps if s.status in ("pending", "failed")]
        lines = [
            "原计划执行时出现失败，请给出修正后的剩余步骤，只输出 JSON 对象。",
            "不要重复已经完成的步骤。",
            "",
            f"任务：{task}",
        ]
        if plan.done:
            lines.append("已完成：" + "；".join(s.description for s in plan.done))
        lines.append("失败原因：" + failure)
        if remaining:
            lines.append("原剩余步骤：" + "；".join(s.description for s in remaining))
        lines.append(f"可用工具：\n{self.tool_catalog or '（无）'}")
        return "\n".join(lines)

    def merge(self, plan: Plan, new_plan: Plan) -> Plan:
        """把重规划结果并回原计划：已完成步骤保留，其余替换。"""
        merged = Plan(goal=plan.goal or new_plan.goal, steps=list(plan.done), revision=plan.revision + 1)
        offset = len(merged.steps)
        for idx, step in enumerate(new_plan.steps, start=1):
            merged.steps.append(
                PlanStep(
                    step_id=offset + idx,
                    description=step.description,
                    tool_hint=step.tool_hint,
                )
            )
        return merged
