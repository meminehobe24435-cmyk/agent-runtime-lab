"""Agent 运行时：规划 → 工具调用 → 观察 → 自我修正 → 终止。

这是整个项目的主循环，也是 JD 职责 1「设计并开发 Agent 框架与运行时」的落点。
运行时负责的事情，刻意与「模型聪不聪明」解耦：

1. **协议层** —— 把工具表按 function calling 格式交给模型，把模型的
   ``tool_calls`` 解码成参数并**在执行前校验**（schema 不合法一律不执行）；
2. **自我修正** —— 参数错、工具抛错、工具名不存在、原地打转，四种失败都有
   明确的回填策略，把「错在哪、正确签名长什么样」交回模型，而不是直接终止；
3. **上下文治理** —— 按 token 预算做**分组压缩**，压缩时保证
   ``assistant(tool_calls)`` 与随后的 ``tool`` 消息同生共死（否则协议就断了）；
4. **可观测性** —— 每一步都写 trace，评测指标直接从 trace 统计，不另立账本。

成功与失败的口径：运行时只判断「是否干净地收敛到最终回答」；
**回答对不对由 :mod:`agentlab.eval` 的评测集判定** —— 两件事分开，指标才不会被污染。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .llm import LLMBackend, Message, ToolCall, messages_tokens
from .memory import ShortTermMemory, WorkingMemory
from .planner import Plan, PlanAndExecutePlanner, ReActPlanner, parse_plan
from .rag import KnowledgeBase
from .tools import ToolRegistry, ToolValidationError
from .trace import (
    PHASE_ACT,
    PHASE_CORRECT,
    PHASE_ERROR,
    PHASE_FINISH,
    PHASE_OBSERVE,
    PHASE_PLAN,
    PHASE_START,
    Tracer,
)


@dataclass
class AgentConfig:
    """运行时可调参数。"""

    max_steps: int = 8
    max_tool_retries: int = 0
    max_identical_calls: int = 2
    max_replans: int = 2
    stop_on_tool_error: bool = False
    context_budget_tokens: int = 4000
    plan_first: bool = False
    rag_top_k: int = 3
    retrieve_before_plan: bool = True
    observation_max_chars: int = 1200
    trace_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class AgentResult:
    """一次运行的完整结果。"""

    task: str
    answer: str
    success: bool
    reason: str = ""
    steps: int = 0
    tool_calls: List[str] = field(default_factory=list)
    successful_tools: List[str] = field(default_factory=list)
    corrections: int = 0
    replans: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    plan: Optional[Plan] = None
    tracer: Optional[Tracer] = None
    working: Optional[WorkingMemory] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "answer": self.answer,
            "success": self.success,
            "reason": self.reason,
            "steps": self.steps,
            "tool_calls": list(self.tool_calls),
            "successful_tools": list(self.successful_tools),
            "corrections": self.corrections,
            "replans": self.replans,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 3),
            "plan": self.plan.to_dict() if self.plan else None,
        }


class AgentRuntime:
    """单智能体运行时。"""

    def __init__(
        self,
        backend: LLMBackend,
        tools: Optional[ToolRegistry] = None,
        *,
        config: Optional[AgentConfig] = None,
        memory: Optional[ShortTermMemory] = None,
        working: Optional[WorkingMemory] = None,
        knowledge: Optional[KnowledgeBase] = None,
        name: str = "agent",
        summarizer: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.backend = backend
        self.tools = tools if tools is not None else ToolRegistry()
        self.config = config or AgentConfig()
        self.memory = memory or ShortTermMemory(max_tokens=self.config.context_budget_tokens)
        self.working = working or WorkingMemory()
        self.knowledge = knowledge
        self.name = name
        self.summarizer = summarizer
        self.runs = 0
        self._planner = ReActPlanner(self.tools.catalog(), self.config.max_steps)
        self._plan_planner = PlanAndExecutePlanner(self.tools.catalog(), self.config.max_steps)

    # ------------------------------------------------------------------ #
    # 提示词
    # ------------------------------------------------------------------ #
    def _system_prompt(self, plan: Optional[Plan] = None, extra: str = "") -> str:
        parts = [
            f"你是 {self.name}，一个可以使用工具解决任务的智能体。",
            "",
            self._planner.system_prompt(),
            "",
            "调用工具时请严格遵守参数的名称、类型与取值范围。",
            "如果工具返回 ERROR，请先阅读错误原因再决定：修正参数重试，或改用别的工具。",
        ]
        if plan is not None and plan.steps:
            parts.extend(["", "当前计划（按顺序推进，可根据观察结果调整）：", plan.render()])
        facts = self.working.render()
        if facts:
            parts.extend(["", facts])
        if extra:
            parts.extend(["", extra])
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def run(self, task: str, context: str = "", plan: Optional[Plan] = None) -> AgentResult:
        """执行一个任务直到收敛或触顶。"""
        started = time.perf_counter()
        cfg = self.config
        tracer = Tracer(name=self.name, enabled=cfg.trace_enabled)
        self.runs += 1
        tracer.event(PHASE_START, f"任务: {task[:80]}", {"task": task, "config": cfg.to_dict()})

        # 1) 可选：先检索资料再规划（RAG 前置）
        rag_block = ""
        if self.knowledge is not None and cfg.retrieve_before_plan:
            hits = self.knowledge.retrieve(task, top_k=cfg.rag_top_k)
            rag_block = self.knowledge.render_context(hits)
            tracer.event(
                PHASE_PLAN,
                f"检索到 {len(hits)} 个片段",
                {"hits": [{"id": h.chunk_id, "score": round(h.score, 4)} for h in hits]},
            )

        if context:
            rag_block = (rag_block + "\n" + context).strip()

        # 2) 可选：先出计划
        current_plan = plan
        if current_plan is None and cfg.plan_first:
            current_plan = self._make_plan(task, rag_block, tracer)

        system = self._system_prompt(current_plan, rag_block)
        messages: List[Message] = [
            Message(role="system", content=system),
            Message(role="user", content=task),
        ]
        self.memory.add("user", task)

        tool_calls_made: List[str] = []
        tools_ok: List[str] = []
        seen_calls: Dict[str, int] = {}
        corrections = 0
        replans = 0
        errors = 0
        answer = ""
        reason = "max_steps_exceeded"
        success = False
        step = 0

        for step in range(1, cfg.max_steps + 1):
            tracer.step_start()

            # 上下文预算治理（属于上下文管理，不计入「自我修正」）
            self._compact(messages, tracer)

            try:
                response = self.backend.chat(messages, self.tools.to_openai())
            except Exception as exc:  # 后端异常不应击穿运行时
                errors += 1
                tracer.event(PHASE_ERROR, f"模型调用失败: {exc}", {"error": type(exc).__name__})
                reason = "backend_error"
                answer = f"模型调用失败：{exc}"
                break

            message = response.message
            messages.append(message)
            if message.content:
                self.memory.add("assistant", message.content)

            # 收敛：没有工具调用就是最终回答
            if not message.tool_calls:
                answer = (message.content or "").strip()
                success = bool(answer)
                reason = "completed" if success else "empty_answer"
                tracer.event(PHASE_FINISH, f"最终回答 {len(answer)} 字", {"reason": reason})
                break

            # 执行工具
            for call in message.tool_calls:
                signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False, default=str)}"

                # 「原地打转」的判据是**成功的重复调用**，而不是调用次数：
                # 工具报错后的重试是有意义的自我修正，不能被当成死循环拦掉。
                if seen_calls.get(signature, 0) >= cfg.max_identical_calls:
                    corrections += 1
                    note = (
                        f"检测到重复调用：{call.name} 已用完全相同的参数成功调用 "
                        f"{seen_calls[signature]} 次。请改用其他方法，或基于已有信息直接给出最终回答。"
                    )
                    tracer.event(PHASE_CORRECT, note, {"tool": call.name, "kind": "loop_detected"})
                    messages.append(
                        Message(role="tool", content=note, tool_call_id=call.id, name=call.name)
                    )
                    continue

                observation, ok, corrected = self._execute(call, step, tracer)
                tool_calls_made.append(call.name)
                if ok:
                    tools_ok.append(call.name)
                    seen_calls[signature] = seen_calls.get(signature, 0) + 1
                else:
                    seen_calls[signature] = 0  # 失败后的重试重新计数
                if corrected:
                    corrections += 1
                if not ok:
                    errors += 1
                    if cfg.stop_on_tool_error:
                        reason = "tool_error"
                        answer = f"工具 {call.name} 执行失败：{observation}"
                        tracer.event(PHASE_ERROR, answer, {"tool": call.name})
                        messages.append(
                            Message(role="tool", content=observation, tool_call_id=call.id, name=call.name)
                        )
                        success = False
                        break
                    # 有计划时：失败一步就重规划剩余路径
                    if current_plan is not None and replans < cfg.max_replans:
                        new_plan = self._replan(task, current_plan, observation, tracer)
                        if new_plan is not None:
                            current_plan = new_plan
                            replans += 1
                            messages[0].content = self._system_prompt(current_plan, rag_block)
                messages.append(
                    Message(role="tool", content=observation, tool_call_id=call.id, name=call.name)
                )
            else:
                continue  # 内层没被 break，正常进入下一轮
            break  # 内层 break（stop_on_tool_error），外层也停

        if not success and reason == "max_steps_exceeded":
            tracer.event(
                PHASE_FINISH,
                f"达到最大步数 {cfg.max_steps} 仍未收敛",
                {"reason": reason, "steps": step},
            )

        duration = (time.perf_counter() - started) * 1000.0
        result = AgentResult(
            task=task,
            answer=answer,
            success=success,
            reason=reason,
            steps=tracer.step,
            tool_calls=tool_calls_made,
            successful_tools=tools_ok,
            corrections=corrections,
            replans=replans,
            errors=errors,
            duration_ms=duration,
            plan=current_plan,
            tracer=tracer,
            working=self.working,
        )
        return result

    # ------------------------------------------------------------------ #
    # 内部：计划 / 工具执行 / 上下文压缩
    # ------------------------------------------------------------------ #
    def _make_plan(self, task: str, context: str, tracer: Tracer) -> Optional[Plan]:
        prompt = self._plan_planner.plan_prompt(task, context)
        try:
            response = self.backend.chat(
                [Message(role="system", content="你是一个任务规划器，只输出 JSON。"),
                 Message(role="user", content=prompt)]
            )
            plan = parse_plan(response.message.content, goal=task)
            tracer.event(PHASE_PLAN, f"生成计划 {len(plan)} 步", {"plan": plan.to_dict()})
            return plan
        except Exception as exc:  # 规划失败降级为 ReAct，不阻断任务
            tracer.event(
                PHASE_CORRECT,
                f"计划解析失败，降级为 ReAct 单步模式: {exc}",
                {"kind": "plan_parse_failed"},
            )
            return None

    def _replan(self, task: str, plan: Plan, failure: str, tracer: Tracer) -> Optional[Plan]:
        """失败后的重规划：保留已完成步骤，只换掉剩余路径。"""
        current = plan.next_step()
        if current is not None:
            plan.mark(current.step_id, "failed", note=failure[:120])
        prompt = self._plan_planner.replan_prompt(task, plan, failure)
        try:
            response = self.backend.chat(
                [
                    Message(role="system", content="你是一个任务规划器，只输出 JSON。"),
                    Message(role="user", content=prompt),
                ]
            )
            fresh = parse_plan(response.message.content, goal=plan.goal or task)
            merged = self._plan_planner.merge(plan, fresh)
            tracer.event(
                PHASE_PLAN,
                f"重规划第 {merged.revision} 版，剩余 {len(merged.pending)} 步",
                {"kind": "replan", "plan": merged.to_dict()},
            )
            return merged
        except Exception as exc:
            tracer.event(
                PHASE_CORRECT,
                f"重规划失败，沿用原计划: {exc}",
                {"kind": "replan_failed"},
            )
            return None

    def _execute(self, call: ToolCall, step: int, tracer: Tracer) -> Tuple[str, bool, bool]:
        """执行一次工具调用。

        返回 ``(observation, ok, corrected)``：``corrected`` 表示这一次是
        「先失败、再纠正」路径（参数非法或工具名错误），用于统计自我修正次数。
        """
        try:
            result = self.tools.call(call.name, call.arguments, max_retries=self.config.max_tool_retries)
        except ToolValidationError as exc:
            # 参数校验失败：把字段级错误 + 正确签名回填给模型
            spec = self.tools.get(call.name)
            detail = (
                f"参数校验失败，工具未执行。\n错误: {'; '.join(exc.errors)}\n"
                f"正确签名: {spec.signature_hint()}"
            )
            tracer.event(
                PHASE_CORRECT,
                f"{call.name} 参数非法: {'; '.join(exc.errors)}",
                {"tool": call.name, "kind": "invalid_arguments", "errors": exc.errors},
            )
            return detail, False, True

        if not result.ok:
            unknown = call.name not in self.tools
            tracer.event(
                PHASE_CORRECT if unknown else PHASE_ACT,
                f"{call.name} 执行失败: {result.error}",
                {
                    "tool": call.name,
                    "ok": False,
                    "error": result.error,
                    "kind": "unknown_tool" if unknown else "tool_error",
                },
            )
            return result.to_observation(self.config.observation_max_chars), False, unknown

        observation = result.to_observation(self.config.observation_max_chars)
        tracer.event(
            PHASE_ACT,
            f"{call.name}({json.dumps(call.arguments, ensure_ascii=False, default=str)[:80]})",
            {"tool": call.name, "ok": True, "arguments": call.arguments,
             "elapsed_ms": result.elapsed_ms},
        )
        tracer.event(PHASE_OBSERVE, observation[:160], {"tool": call.name})
        # 成功结果写进工作记忆，带来源溯源（结构化结果统一序列化后截断存）
        self.working.set(f"last_{call.name}", result.value, source=f"tool:{call.name}", step=step)
        return observation, True, False

    def _compact(self, messages: List[Message], tracer: Tracer) -> bool:
        """按 token 预算压缩上下文，返回是否发生了压缩。

        压缩以「消息组」为单位：``assistant(tool_calls)`` 与它后面所有的
        ``tool`` 消息必须一起保留或一起丢弃，否则函数调用协议不再成对。
        """
        budget = self.config.context_budget_tokens
        if messages_tokens(messages) <= budget:
            return False

        system, rest = messages[0], messages[1:]
        groups: List[List[Message]] = []
        for msg in rest:
            if msg.role == "tool" and groups and any(c for c in groups[-1] if c.tool_calls):
                groups[-1].append(msg)
            else:
                groups.append([msg])

        dropped: List[Message] = []
        while len(groups) > 1 and messages_tokens([system] + [m for g in groups for m in g]) > budget:
            dropped.extend(groups.pop(0))
        if not dropped:
            return False

        digest = self._summarize(dropped)
        if digest:
            system.content = system.content + f"\n\n[已压缩的历史]\n{digest}"
        messages[:] = [system] + [m for g in groups for m in g]
        tracer.event(
            PHASE_CORRECT,
            f"上下文压缩：折叠 {len(dropped)} 条历史消息",
            {"kind": "context_compaction", "dropped": len(dropped)},
        )
        return True

    def _summarize(self, dropped: Sequence[Message]) -> str:
        text = "\n".join(
            f"{m.role}: {(m.content or '').strip()}" for m in dropped if (m.content or "").strip()
        )
        if not text:
            return ""
        if self.summarizer is not None:
            try:
                return self.summarizer(text)
            except Exception:
                pass
        return text[:600]
