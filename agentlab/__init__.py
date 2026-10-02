"""agentlab —— 一个零第三方依赖的 LLM Agent 运行时与实践台。

分层（从下到上）：

- :mod:`agentlab.llm`          模型后端抽象（脚本化 / 规则化 / OpenAI 兼容）
- :mod:`agentlab.tools`        工具注册、JSON Schema 校验、Function Calling 协议
- :mod:`agentlab.memory`       短期缓冲、长期向量记忆、工作记忆
- :mod:`agentlab.rag`          切块、检索、引用溯源（RAG-as-a-tool）
- :mod:`agentlab.planner`      提示词构造与计划解析（ReAct / Plan-and-Execute）
- :mod:`agentlab.runtime`      Agent 主循环：规划 → 工具调用 → 观察 → 自我修正
- :mod:`agentlab.orchestrator` Supervisor + Worker + 消息总线（多智能体）
- :mod:`agentlab.trace`        结构化 trace（可观测性）
- :mod:`agentlab.eval`         任务集、指标、报告（评测体系）

设计原则：**运行时不依赖任何 SDK，也不依赖网络** ——
真实模型通过 :class:`~agentlab.llm.OpenAICompatBackend` 接入，
测试与评测则用确定性的脚本后端，所以整套东西在 CI 里可以完整复现。
"""

from .eval import (
    EvalReport,
    EvalTask,
    SuiteRunner,
    TaskOutcome,
    build_knowledge_base,
    build_ops_tools,
    build_suite_tasks,
    make_scenario_factory,
    run_default_suite,
)
from .llm import (
    ChatResponse,
    LLMBackend,
    Message,
    OpenAICompatBackend,
    RuleBasedBackend,
    ScriptedBackend,
    ToolCall,
    estimate_tokens,
)
from .memory import Fact, ShortTermMemory, VectorMemory, WorkingMemory, cosine, embed
from .orchestrator import (
    BusMessage,
    MessageBus,
    OrchestrationResult,
    SubTask,
    Supervisor,
    Worker,
    default_decompose,
)
from .planner import (
    Plan,
    PlanAndExecutePlanner,
    PlanParseError,
    PlanStep,
    ReActPlanner,
    load_json_block,
    parse_decision,
    parse_plan,
)
from .rag import Chunk, Hit, KnowledgeBase, build_search_tool, chunk_text
from .runtime import AgentConfig, AgentResult, AgentRuntime
from .tools import (
    FileSandbox,
    ToolError,
    ToolNotFound,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolValidationError,
    build_file_tools,
    calculator,
    default_registry,
    safe_eval,
    tool,
    validate_arguments,
)
from .trace import (
    PHASE_ACT,
    PHASE_CORRECT,
    PHASE_DELEGATE,
    PHASE_ERROR,
    PHASE_FINISH,
    PHASE_OBSERVE,
    PHASE_PLAN,
    PHASE_START,
    TraceEvent,
    Tracer,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # llm
    "ChatResponse", "LLMBackend", "Message", "OpenAICompatBackend", "RuleBasedBackend",
    "ScriptedBackend", "ToolCall", "estimate_tokens",
    # tools
    "FileSandbox", "ToolError", "ToolNotFound", "ToolRegistry", "ToolResult", "ToolSpec",
    "ToolValidationError", "build_file_tools", "calculator", "default_registry", "safe_eval",
    "tool", "validate_arguments",
    # memory
    "Fact", "ShortTermMemory", "VectorMemory", "WorkingMemory", "cosine", "embed",
    # rag
    "Chunk", "Hit", "KnowledgeBase", "build_search_tool", "chunk_text",
    # planner
    "Plan", "PlanAndExecutePlanner", "PlanParseError", "PlanStep", "ReActPlanner",
    "load_json_block", "parse_decision", "parse_plan",
    # runtime
    "AgentConfig", "AgentResult", "AgentRuntime",
    # orchestrator
    "BusMessage", "MessageBus", "OrchestrationResult", "SubTask", "Supervisor", "Worker",
    "default_decompose",
    # trace
    "PHASE_ACT", "PHASE_CORRECT", "PHASE_DELEGATE", "PHASE_ERROR", "PHASE_FINISH",
    "PHASE_OBSERVE", "PHASE_PLAN", "PHASE_START", "TraceEvent", "Tracer",
    # eval
    "EvalReport", "EvalTask", "SuiteRunner", "TaskOutcome", "build_knowledge_base",
    "build_ops_tools", "build_suite_tasks", "make_scenario_factory", "run_default_suite",
]
