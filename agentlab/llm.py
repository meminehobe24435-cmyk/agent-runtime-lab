"""LLM 后端抽象层。

设计目标：让 Agent 运行时不依赖任何第三方 SDK，并且**在没有网络、没有 API Key 的
CI 环境里也能完整跑通**。

- ``LLMBackend``            统一接口，只有 ``chat(messages, tools)`` 一个方法
- ``ScriptedBackend``       按脚本依次返回预设回复，确定性、零网络（单元测试用）
- ``RuleBasedBackend``      按可用工具与观察结果做规则决策（端到端评测用）
- ``OpenAICompatBackend``   通过 HTTP 调用 OpenAI 兼容的 /chat/completions
                            （DeepSeek / 通义 / vLLM 等），需要 API Key

消息格式刻意与 OpenAI 的 function calling 协议保持一致：
assistant 消息可携带 ``tool_calls``，工具结果以 ``role="tool"`` + ``tool_call_id`` 回填。
这样运行时的核心逻辑可以直接对接真实模型，不需要在中间做一层格式翻译。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# 消息与工具调用
# --------------------------------------------------------------------------- #

@dataclass
class ToolCall:
    """一次函数调用请求（对应 OpenAI 的 tool_calls 元素）。"""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    def to_wire(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_wire(cls, obj: Dict[str, Any]) -> "ToolCall":
        fn = obj.get("function") or {}
        raw = fn.get("arguments")
        args: Dict[str, Any] = {}
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
        return cls(
            id=str(obj.get("id") or ""),
            name=str(fn.get("name") or ""),
            arguments=args,
            raw_arguments=raw if isinstance(raw, str) else "",
        )


@dataclass
class Message:
    """一条对话消息。``role`` ∈ {system, user, assistant, tool}。"""

    role: str
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        wire: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            wire["tool_calls"] = [c.to_wire() for c in self.tool_calls]
        if self.tool_call_id is not None:
            wire["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            wire["name"] = self.name
        return wire

    @classmethod
    def from_wire(cls, obj: Dict[str, Any]) -> "Message":
        return cls(
            role=str(obj.get("role") or "assistant"),
            content=str(obj.get("content") or ""),
            tool_calls=[ToolCall.from_wire(c) for c in (obj.get("tool_calls") or [])],
            tool_call_id=obj.get("tool_call_id"),
            name=obj.get("name"),
        )


@dataclass
class ChatResponse:
    """后端返回的一条 assistant 消息 + 可选的用量统计。"""

    message: Message
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def text(self) -> str:
        return self.message.content


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：CJK 按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token。

    只是给「上下文预算」用的保守估计，不追求与具体 tokenizer 对齐。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def messages_tokens(messages: Sequence[Message]) -> int:
    return sum(estimate_tokens(m.content) for m in messages)


# --------------------------------------------------------------------------- #
# 后端接口
# --------------------------------------------------------------------------- #

class LLMBackend:
    """后端基类。子类只需实现 :meth:`chat`。"""

    name = "base"

    def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ChatResponse:  # pragma: no cover - 抽象方法
        raise NotImplementedError

    # 便于测试：记录调用历史
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []


class ScriptedBackend(LLMBackend):
    """按脚本依次返回预设回复的确定性后端。

    ``script`` 的每个元素可以是：

    - ``str``                     → 直接作为最终回答（无工具调用）
    - ``dict``                    → ``{"content": ..., "tool_calls": [...]}``
    - ``Message``                 → 原样返回
    - ``Exception``               → 抛出（用于测试后端异常路径）

    ``tool_calls`` 支持简写 ``{"name": ..., "arguments": {...}}``，
    ``id`` 会自动补成 ``call_1``、``call_2`` ……
    """

    name = "scripted"

    def __init__(self, script: Iterable[Any], repeat_last: bool = False) -> None:
        super().__init__()
        self.script: List[Any] = list(script)
        self.repeat_last = repeat_last
        self._index = 0
        self._call_seq = 0

    def _coerce(self, item: Any) -> Message:
        if isinstance(item, Message):
            return item
        if isinstance(item, str):
            return Message(role="assistant", content=item)
        if isinstance(item, dict):
            calls: List[ToolCall] = []
            for raw in item.get("tool_calls") or []:
                self._call_seq += 1
                if isinstance(raw, ToolCall):
                    calls.append(raw)
                    continue
                name = raw.get("name") or ""
                args = raw.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                calls.append(
                    ToolCall(
                        id=str(raw.get("id") or f"call_{self._call_seq}"),
                        name=name,
                        arguments=dict(args),
                    )
                )
            return Message(
                role="assistant",
                content=str(item.get("content") or ""),
                tool_calls=calls,
            )
        raise TypeError(f"不支持的脚本元素: {type(item)!r}")

    def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        if self._index >= len(self.script):
            if not self.repeat_last or not self.script:
                return ChatResponse(
                    message=Message(role="assistant", content="[scripted] 脚本已耗尽"),
                    latency_ms=0.0,
                )
            item = self.script[-1]
        else:
            item = self.script[self._index]
            self._index += 1
        if isinstance(item, Exception):
            raise item
        return ChatResponse(message=self._coerce(item), latency_ms=0.0)


class RuleBasedBackend(LLMBackend):
    """规则驱动的确定性后端，用于端到端评测。

    它的行为**不是**「假模型」，而是把「一个有能力的 Agent 应当怎么做」写成显式规则：
    先看任务里要解决的问题类型，选择合适的工具，拿到观察结果后判断是否还需要下一步。
    这样可以给运行时与评测体系提供**可重复、可归因**的端到端流量，
    而不需要把不确定性引入 CI。

    规则由调用方以 ``rules`` 注入，每条规则形如::

        {"match": ("温度", "告警"), "tool": "query_metric", "arguments": {...},
         "until": "observed", "answer": "……"}

    - ``match``   任务文本里全部命中的关键词
    - ``tool``    要调用的工具；为 ``None`` 时直接以 ``answer`` 结束
    - ``until``   ``"always"`` = 每次都调用；``"observed"`` = 只在还没有该工具的
                  观察结果时调用一次（模拟「拿到信息就不再重复取数」）
    - ``attempts`` 前 N 次故意给出错误参数（用于触发自我修正路径）
    - ``retry_on_error`` 上一次该工具返回 ERROR 时是否允许重试（默认允许 1 次）
    """

    name = "rule-based"

    def __init__(self, rules: Sequence[Dict[str, Any]]) -> None:
        super().__init__()
        self.rules = [dict(r) for r in rules]
        self._call_seq = 0
        self._attempts: Dict[str, int] = {}
        self._error_retries: Dict[str, int] = {}

    def _last_error(self, messages: Sequence[Message], tool: str) -> bool:
        for msg in reversed(messages):
            if msg.role == "tool" and msg.name == tool:
                return (msg.content or "").lstrip().startswith("ERROR")
        return False

    @staticmethod
    def _observed(messages: Sequence[Message]) -> List[str]:
        return [str(m.name) for m in messages if m.role == "tool"]

    def _pick(self, task_text: str) -> Optional[Dict[str, Any]]:
        for rule in self.rules:
            keys = rule.get("match") or ()
            if keys and all(k in task_text for k in keys):
                return rule
        return None

    def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        task_text = " ".join(m.content for m in messages if m.role == "user")
        tool_names = {t["function"]["name"] for t in (tools or [])}
        observed = self._observed(messages)
        rule = self._pick(task_text)

        if rule is None:
            return ChatResponse(
                message=Message(role="assistant", content="没有匹配到可执行规则，无法完成该任务。")
            )

        tool = rule.get("tool")
        if not tool:
            return ChatResponse(
                message=Message(role="assistant", content=str(rule.get("answer") or "已完成。"))
            )

        if tool not in tool_names:
            return ChatResponse(
                message=Message(
                    role="assistant",
                    content=f"缺少完成该任务所需的工具 {tool}。",
                )
            )

        if rule.get("until") == "observed" and tool in observed:
            retry_allowed = bool(rule.get("retry_on_error", True))
            limit = int(rule.get("max_error_retries", 1))
            if retry_allowed and self._last_error(messages, tool) and self._error_retries.get(tool, 0) < limit:
                self._error_retries[tool] = self._error_retries.get(tool, 0) + 1
            else:
                return ChatResponse(
                    message=Message(
                        role="assistant", content=str(rule.get("answer") or "已根据观察结果完成。")
                    )
                )

        self._call_seq += 1
        args = dict(rule.get("arguments") or {})
        # 需要构造一次失败：前 attempts 次给出被污染的参数，触发运行时的自我修正
        bad = int(rule.get("attempts") or 0)
        used = self._attempts.get(tool, 0)
        if used < bad:
            self._attempts[tool] = used + 1
            args = dict(rule.get("bad_arguments") or args)
        return ChatResponse(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id=f"call_{self._call_seq}", name=tool, arguments=args)
                ],
            )
        )


class OpenAICompatBackend(LLMBackend):
    """OpenAI 兼容的 HTTP 后端（DeepSeek / OpenAI / vLLM / Ollama 均可）。

    仅用标准库 ``urllib`` 实现，避免为了跑通 Agent 而引入 SDK 依赖。
    CI 中不会实例化该类（没有 API Key），它服务于本地真机联调。
    """

    name = "openai-compat"

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__()
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("AGENTLAB_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.deepseek.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("AGENTLAB_API_KEY") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_wire() for m in messages],
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        return payload

    def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        if not self.available:
            raise RuntimeError(
                "未配置 API Key（AGENTLAB_API_KEY / OPENAI_API_KEY），无法调用真实模型。"
            )
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        body = json.dumps(self._payload(messages, tools)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                latency = (time.perf_counter() - started) * 1000.0
                choice = (data.get("choices") or [{}])[0]
                message = Message.from_wire(choice.get("message") or {})
                usage = data.get("usage") or {}
                return ChatResponse(
                    message=message,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency_ms=latency,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"调用模型失败（已重试 {self.max_retries} 次）: {last_error}")
