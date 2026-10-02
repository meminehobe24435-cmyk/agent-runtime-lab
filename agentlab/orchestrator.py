"""多智能体协作：Supervisor + Worker + 消息总线。

JD 职责 1 里「多智能体协作」在工程上要解决三个具体问题，本模块就按这三个来设计：

1. **任务怎么分** —— :class:`Supervisor` 把大任务拆成子任务（可以由规则拆，
   也可以让模型出计划），再按**技能标签与可用工具**给子任务选一个 Worker；
2. **子任务失败怎么办** —— 允许把失败的子任务**改派给次优 Worker**（换工具重试），
   超过重试上限才判定该子任务失败，并在汇总里如实标注；
3. **信息怎么流转** —— :class:`MessageBus` 记录每一次委派与回执（谁在做、做了什么、
   耗时多久、结果如何），最终汇总时按**已完成的子任务**作答，不把没跑成的部分说成跑成了。

子 Agent 之间**不共享对话历史**，只通过消息总线交换结构化回执 ——
这样单个 Worker 的上下文不会被别的子任务污染，也让「谁说了什么」可追溯。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .llm import LLMBackend, Message
from .memory import tokenize
from .runtime import AgentConfig, AgentResult, AgentRuntime
from .trace import PHASE_DELEGATE, PHASE_ERROR, PHASE_FINISH, PHASE_START, Tracer


# --------------------------------------------------------------------------- #
# 消息总线
# --------------------------------------------------------------------------- #

@dataclass
class BusMessage:
    sender: str
    recipient: str
    kind: str  # delegate / result / error / summary
    payload: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "kind": self.kind,
            "payload": self.payload,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class MessageBus:
    """极简消息总线：发布 + 订阅 + 留痕。"""

    def __init__(self) -> None:
        self.history: List[BusMessage] = []
        self._subs: Dict[str, List[Callable[[BusMessage], None]]] = {}

    def subscribe(self, kind: str, handler: Callable[[BusMessage], None]) -> None:
        self._subs.setdefault(kind, []).append(handler)

    def publish(self, message: BusMessage) -> BusMessage:
        self.history.append(message)
        for handler in self._subs.get(message.kind, []):
            try:
                handler(message)
            except Exception:  # 订阅方异常不影响主流程
                continue
        return message

    def of_kind(self, kind: str) -> List[BusMessage]:
        return [m for m in self.history if m.kind == kind]

    def stats(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for msg in self.history:
            counts[msg.kind] = counts.get(msg.kind, 0) + 1
        return {"messages": len(self.history), "by_kind": counts}


# --------------------------------------------------------------------------- #
# 子任务与 Worker
# --------------------------------------------------------------------------- #

@dataclass
class SubTask:
    task_id: int
    description: str
    skills: List[str] = field(default_factory=list)
    status: str = "pending"  # pending / done / failed
    assignee: str = ""
    answer: str = ""
    attempts: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.task_id,
            "description": self.description,
            "skills": list(self.skills),
            "status": self.status,
            "assignee": self.assignee,
            "attempts": self.attempts,
            "answer": self.answer,
            "error": self.error,
        }


@dataclass
class Worker:
    """一个具备特定技能子集的子 Agent。"""

    name: str
    runtime: AgentRuntime
    skills: Sequence[str] = ()
    description: str = ""

    def score(self, text: str, skill_hint: Sequence[str] = ()) -> float:
        """给子任务打分：技能命中数 + 工具名命中数（都做大小写无关的包含匹配）。"""
        lowered = text.lower()
        score = 0.0
        for skill in list(skill_hint) + list(self.skills):
            if skill and skill.lower() in lowered:
                score += 3.0
        for tool_name in self.runtime.tools.names:
            if tool_name.lower() in lowered:
                score += 2.0
        if not self.skills and not skill_hint:
            score += 0.5  # 无技能声明的 Worker 作为兜底候选
        return score

    def accepts(self, result: AgentResult) -> bool:
        """**子任务验收判据**（决定要不要改派给别的 Worker）。

        运行时的 ``success`` 只表示「干净地收敛到了最终回答」，
        它不判断回答有没有依据。放到多智能体场景里就会出现一种危险情况：
        Worker 的每个工具都失败了，模型仍然"基于常识"给出一段回答 ——
        运行时判成功，而子任务其实什么也没查到。
        所以这里追加一条：**调过工具却一次都没成功的，不算完成**。
        """
        if not result.success:
            return False
        if result.tool_calls and not result.successful_tools:
            return False
        return True

    def run(self, task: str, context: str = "") -> AgentResult:
        return self.runtime.run(task, context=context)


@dataclass
class OrchestrationResult:
    task: str
    answer: str
    success: bool
    subtasks: List[SubTask] = field(default_factory=list)
    worker_results: Dict[int, AgentResult] = field(default_factory=dict)
    rounds: int = 0
    duration_ms: float = 0.0
    tracer: Optional[Tracer] = None
    bus: Optional[MessageBus] = None
    reason: str = ""

    @property
    def completed(self) -> List[SubTask]:
        return [s for s in self.subtasks if s.status == "done"]

    @property
    def failed(self) -> List[SubTask]:
        return [s for s in self.subtasks if s.status == "failed"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "answer": self.answer,
            "success": self.success,
            "reason": self.reason,
            "rounds": self.rounds,
            "duration_ms": round(self.duration_ms, 3),
            "subtasks": [s.to_dict() for s in self.subtasks],
            "bus": self.bus.stats() if self.bus else {},
        }


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #

_SPLIT_MARKERS = (
    "，然后", "；然后", "，接着", "，最后", "，并且", "，以及",
    "然后", "接着", "最后", "并且", "以及",
    "；", "。", "\n", ",",
)


def _meaningful(piece: str) -> bool:
    """判断一个切分片段是不是「一个真任务」。

    中文 3 个字就可能是完整指令（「查温度」），而英文要 4 个字符才算；
    统一按字符数过滤会把中文短任务整段丢掉。
    """
    cjk = sum(1 for ch in piece if "\u4e00" <= ch <= "\u9fff")
    return cjk >= 2 or len(piece) >= 4


def default_decompose(task: str, max_parts: int = 4) -> List[SubTask]:
    """默认拆分策略：按连接词/标点切分，过滤掉「好」「嗯」这类碎片。

    这不是「智能拆分」，但对「先查 A 再算 B 顺便写 C」这类串行任务足够，
    而且**完全确定**，评测时不会因为拆分不稳定导致指标抖动。
    """
    text = (task or "").strip()
    if not text:
        return []
    parts = [text]
    for marker in _SPLIT_MARKERS:
        expanded: List[str] = []
        for piece in parts:
            expanded.extend(piece.split(marker))
        parts = expanded
    cleaned = [p.strip(" ，,。;；\n\t") for p in parts]
    cleaned = [p for p in cleaned if _meaningful(p)]
    if not cleaned:
        cleaned = [text]
    # 去重但保序
    seen: set = set()
    unique: List[str] = []
    for piece in cleaned:
        if piece not in seen:
            seen.add(piece)
            unique.append(piece)
    return [SubTask(task_id=i, description=p) for i, p in enumerate(unique[:max_parts], start=1)]


class Supervisor:
    """把任务拆给多个 Worker，并汇总结果。"""

    def __init__(
        self,
        workers: Sequence[Worker],
        *,
        backend: Optional[LLMBackend] = None,
        bus: Optional[MessageBus] = None,
        decomposer: Optional[Callable[[str], List[SubTask]]] = None,
        synthesized: bool = True,
        max_parts: int = 4,
        max_attempts_per_task: int = 2,
        name: str = "supervisor",
        config: Optional[AgentConfig] = None,
    ) -> None:
        if not workers:
            raise ValueError("Supervisor 至少需要一个 Worker")
        self.workers = list(workers)
        self.backend = backend
        self.bus = bus or MessageBus()
        self.decomposer = decomposer or (lambda t: default_decompose(t, max_parts))
        self.synthesized = synthesized
        self.max_attempts_per_task = max_attempts_per_task
        self.name = name
        self.config = config or AgentConfig()
        self.delegations = 0

    # -- 选择 Worker ------------------------------------------------------ #
    def rank(self, subtask: SubTask) -> List[Worker]:
        """按匹配度排序候选 Worker；同分时保持注册顺序，保证结果可复现。"""
        text = " ".join([subtask.description] + list(subtask.skills))
        scored = [(w.score(text, subtask.skills), idx, w) for idx, w in enumerate(self.workers)]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [w for _, _, w in scored]

    # -- 汇总 ------------------------------------------------------------- #
    def aggregate(self, task: str, subtasks: Sequence[SubTask]) -> str:
        done = [s for s in subtasks if s.status == "done"]
        failed = [s for s in subtasks if s.status == "failed"]
        lines = [f"任务：{task}", f"完成 {len(done)}/{len(subtasks)} 个子任务。"]
        for sub in done:
            lines.append(f"[{sub.task_id}] ({sub.assignee}) {sub.description} → {sub.answer}")
        for sub in failed:
            lines.append(f"[{sub.task_id}] 未完成：{sub.description}（原因：{sub.error}）")
        return "\n".join(lines)

    def _synthesize(self, task: str, subtasks: Sequence[SubTask]) -> Optional[str]:
        """让模型把子任务结果整合成一段自然语言回答（无后端时返回 None）。"""
        if self.backend is None:
            return None
        payload = "\n".join(
            f"- {s.description} => {s.answer}" for s in subtasks if s.status == "done"
        )
        if not payload:
            return None
        try:
            response = self.backend.chat(
                [
                    Message(
                        role="system",
                        content="你是汇总者。只根据给定的子任务结论作答，不要引入新的事实。",
                    ),
                    Message(role="user", content=f"总任务：{task}\n子任务结论：\n{payload}\n请给出最终回答。"),
                ]
            )
            text = (response.message.content or "").strip()
            return text or None
        except Exception:
            return None

    # -- 主流程 ----------------------------------------------------------- #
    def run(self, task: str, context: str = "") -> OrchestrationResult:
        started = time.perf_counter()
        tracer = Tracer(name=self.name, enabled=self.config.trace_enabled)
        tracer.event(PHASE_START, f"总任务: {task[:80]}", {"workers": [w.name for w in self.workers]})

        subtasks = self.decomposer(task)
        if not subtasks:
            subtasks = [SubTask(task_id=1, description=task)]
        for sub in subtasks:
            tracer.step_start()
            tracer.event(
                PHASE_DELEGATE,
                f"子任务 {sub.task_id}: {sub.description[:60]}",
                {"subtask": sub.to_dict()},
            )

        results: Dict[int, AgentResult] = {}
        rounds = 0
        for sub in subtasks:
            candidates = self.rank(sub)
            for attempt, worker in enumerate(candidates[: self.max_attempts_per_task], start=1):
                self.delegations += 1
                sub.attempts = attempt
                sub.assignee = worker.name
                self.bus.publish(
                    BusMessage(
                        sender=self.name,
                        recipient=worker.name,
                        kind="delegate",
                        payload={"task_id": sub.task_id, "description": sub.description,
                                 "attempt": attempt},
                    )
                )
                begin = time.perf_counter()
                try:
                    result = worker.run(sub.description, context=context)
                    error = ""
                except Exception as exc:  # Worker 崩溃 → 换人重试
                    result = AgentResult(
                        task=sub.description, answer="", success=False,
                        reason="worker_exception",
                    )
                    error = f"{type(exc).__name__}: {exc}"

                elapsed = (time.perf_counter() - begin) * 1000.0
                rounds += 1
                results[sub.task_id] = result
                ok = worker.accepts(result)
                if not ok and not error:
                    error = result.reason if not result.success else "工具全部失败，未取得有效结果"
                self.bus.publish(
                    BusMessage(
                        sender=worker.name,
                        recipient=self.name,
                        kind="result" if ok else "error",
                        payload={
                            "task_id": sub.task_id,
                            "success": ok,
                            "answer": result.answer[:400],
                            "error": error or result.reason,
                            "steps": result.steps,
                        },
                        elapsed_ms=elapsed,
                    )
                )
                tracer.event(
                    PHASE_DELEGATE if ok else PHASE_ERROR,
                    f"{worker.name} {'完成' if ok else '失败'} 子任务 {sub.task_id}",
                    {
                        "subtask_id": sub.task_id,
                        "worker": worker.name,
                        "attempt": attempt,
                        "ok": ok,
                        "elapsed_ms": round(elapsed, 3),
                        "error": error or result.reason,
                    },
                )
                if ok:
                    sub.status = "done"
                    sub.answer = result.answer
                    sub.error = ""
                    break
                sub.error = error or result.reason or "未知失败"
            else:
                sub.status = "failed"

        answer = None
        if self.synthesized:
            answer = self._synthesize(task, subtasks)
        if not answer:
            answer = self.aggregate(task, subtasks)

        success = bool(subtasks) and all(s.status == "done" for s in subtasks)
        reason = "completed" if success else ("partial" if any(s.status == "done" for s in subtasks) else "failed")
        tracer.event(PHASE_FINISH, f"{len([s for s in subtasks if s.status == 'done'])}/{len(subtasks)} 子任务完成",
                     {"reason": reason})

        return OrchestrationResult(
            task=task,
            answer=answer,
            success=success,
            subtasks=subtasks,
            worker_results=results,
            rounds=rounds,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            tracer=tracer,
            bus=self.bus,
            reason=reason,
        )
