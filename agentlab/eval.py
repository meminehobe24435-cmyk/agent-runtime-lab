"""评测体系：任务集、指标、报告。

对应 JD 职责 4「建立 Agent 效果评测体系，通过数据分析与迭代持续优化任务完成质量」。
这里刻意把「评测」和「跑一遍」区分开：

- **任务集**（:class:`EvalTask`）写清每个任务的**期望工具**、**期望答案要点**、
  步数上限与禁用工具 —— 期望值是人工写死的，不是把实际输出抄回来；
- **判据**（:meth:`EvalTask.check`）逐项给出通过与否，报告里能看到**为什么挂**，
  而不是只看到一个总分；
- **指标**（:class:`EvalReport`）分四组：任务完成、工具使用、自我修正、成本，
  其中「自我修正恢复率」= 出过错但最终仍然通过的任务数 ÷ 出过错误的任务数 ——
  这是衡量运行时韧性最直接的一个数。

> ⚠️ **口径说明（写进 README 与简历的边界）**：本评测集使用**脚本化模型后端**
> （:class:`~agentlab.llm.ScriptedBackend`）产生确定性的工具调用序列，
> 目的是让「运行时 / 工具层 / 记忆 / 自我修正」这些**确定性组件**的缺陷能被稳定复现，
> 而不是衡量某个具体大模型的能力。换成真实模型时，**同一套任务、判据与指标可直接复用**。
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .llm import ScriptedBackend
from .memory import ShortTermMemory, VectorMemory, WorkingMemory
from .rag import KnowledgeBase, build_search_tool
from .runtime import AgentConfig, AgentResult, AgentRuntime
from .tools import FileSandbox, ToolRegistry, build_file_tools, calculator
from .trace import PHASE_ACT, PHASE_CORRECT


# --------------------------------------------------------------------------- #
# 任务定义
# --------------------------------------------------------------------------- #

@dataclass
class EvalTask:
    """一个评测任务。"""

    id: str
    prompt: str
    script: List[Any] = field(default_factory=list)
    expected_tools: Tuple[str, ...] = ()
    expected_answer_contains: Tuple[str, ...] = ()
    expected_sequence: Optional[Tuple[str, ...]] = None
    forbidden_tools: Tuple[str, ...] = ()
    max_steps: int = 8
    expect_recovery: bool = False
    plan_first: bool = False
    context_budget_tokens: Optional[int] = None
    expect_replans: int = 0
    expect_compaction: bool = False
    tags: Tuple[str, ...] = ()
    note: str = ""

    def check(self, result: AgentResult) -> Dict[str, bool]:
        answer = result.answer or ""
        all_calls = list(result.tool_calls)
        ok_calls = list(result.successful_tools)
        checks = {
            "收敛": result.success,
            "答案要点": all(k in answer for k in self.expected_answer_contains),
            "必需工具": all(t in ok_calls for t in self.expected_tools),
            "未用禁用工具": not any(t in ok_calls for t in self.forbidden_tools),
            "未超步数": result.steps <= self.max_steps,
        }
        if self.expected_sequence is not None:
            checks["工具序列严格匹配"] = tuple(all_calls) == tuple(self.expected_sequence)
        if self.expect_replans:
            checks["发生重规划"] = result.replans >= self.expect_replans
        if self.expect_compaction:
            checks["触发上下文压缩"] = any(
                e.data.get("kind") == "context_compaction" for e in result.tracer.events
            )
        return checks


@dataclass
class TaskOutcome:
    task: EvalTask
    result: Optional[AgentResult]
    checks: Dict[str, bool] = field(default_factory=dict)
    error: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def failed_checks(self) -> List[str]:
        return [k for k, v in self.checks.items() if not v]

    @property
    def tool_calls(self) -> List[str]:
        return list(self.result.tool_calls) if self.result else []

    @property
    def successful_tools(self) -> List[str]:
        return list(self.result.successful_tools) if self.result else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.task.id,
            "passed": self.passed,
            "failed_checks": self.failed_checks,
            "checks": self.checks,
            "tool_calls": self.tool_calls,
            "successful_tools": self.successful_tools,
            "steps": self.result.steps if self.result else 0,
            "corrections": self.result.corrections if self.result else 0,
            "replans": self.result.replans if self.result else 0,
            "errors": self.result.errors if self.result else 0,
            "duration_ms": round(self.result.duration_ms, 3) if self.result else 0.0,
            "answer": (self.result.answer[:200] if self.result else ""),
            "error": self.error,
            "tags": list(self.task.tags),
        }


# --------------------------------------------------------------------------- #
# 报告与指标
# --------------------------------------------------------------------------- #

@dataclass
class EvalReport:
    outcomes: List[TaskOutcome] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    label: str = "suite"

    # -- 基础统计 --------------------------------------------------------- #
    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    def metrics(self) -> Dict[str, Any]:
        outcomes = self.outcomes
        total = len(outcomes) or 1
        calls: List[str] = []
        distinct: List[str] = []
        for outcome in outcomes:
            got = outcome.tool_calls
            calls.extend(got)
            distinct.extend(sorted(set(got)))

        tool_recall: List[float] = []
        tool_precision: List[float] = []
        for outcome in outcomes:
            expected = set(outcome.task.expected_tools)
            actual = set(outcome.successful_tools)
            if expected:
                tool_recall.append(len(expected & actual) / len(expected))
            if actual or expected:
                tool_precision.append(len(expected & actual) / len(actual) if actual else 0.0)

        errored = [o for o in outcomes if o.result and o.result.errors > 0]
        recovered = [o for o in errored if o.passed]
        corrected = [o for o in outcomes if o.result and o.result.corrections > 0]
        replanned = [o for o in outcomes if o.result and o.result.replans > 0]

        durations = [o.result.duration_ms for o in outcomes if o.result]
        steps = [o.result.steps for o in outcomes if o.result]

        def _mean(values: Sequence[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        return {
            "任务数": len(outcomes),
            "通过": self.passed,
            "任务成功率": round(self.passed / total, 4),
            "工具召回率": _mean(tool_recall),
            "工具准确率": _mean(tool_precision),
            "工具调用总数": len(calls),
            "冗余调用": len(calls) - len(distinct),
            "平均步数": _mean(steps),
            "最大步数": max(steps) if steps else 0,
            "出错任务数": len(errored),
            "自我修正恢复率": round(len(recovered) / len(errored), 4) if errored else 0.0,
            "发生自我修正的任务": len(corrected),
            "发生重规划的任务": len(replanned),
            "参数校验纠正次数": sum(
                1
                for o in outcomes
                if o.result
                for e in o.result.tracer.corrections()
                if e.data.get("kind") == "invalid_arguments"
            ),
            "未知工具纠正次数": sum(
                1
                for o in outcomes
                if o.result
                for e in o.result.tracer.corrections()
                if e.data.get("kind") == "unknown_tool"
            ),
            "死循环拦截次数": sum(
                1
                for o in outcomes
                if o.result
                for e in o.result.tracer.corrections()
                if e.data.get("kind") == "loop_detected"
            ),
            "平均耗时ms": _mean(durations),
            "中位耗时ms": round(statistics.median(durations), 4) if durations else 0.0,
            "评测耗时ms": round(self.duration_ms, 3),
        }

    # -- 导出 ------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "metrics": self.metrics(),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: Any) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def to_markdown(self) -> str:
        m = self.metrics()
        lines = [
            f"# Agent 评测报告（{self.label}）",
            "",
            "## 总体指标",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
        ]
        for key, value in m.items():
            lines.append(f"| {key} | {value} |")
        lines.extend(
            [
                "",
                "## 逐任务明细",
                "",
                "| 任务 | 通过 | 失败项 | 步数 | 自我修正 | 工具调用 | 耗时(ms) |",
                "|---|---|---|---:|---:|---|---:|",
            ]
        )
        for out in self.outcomes:
            marks = "、".join(out.failed_checks) or "-"
            tools = " → ".join(out.tool_calls) or "-"
            lines.append(
                f"| {out.task.id} | {'✅' if out.passed else '❌'} | {marks} | "
                f"{out.result.steps if out.result else 0} | "
                f"{out.result.corrections if out.result else 0} | {tools} | "
                f"{out.result.duration_ms if out.result else 0:.1f} |"
            )
        lines.append("")
        lines.append("> 评测使用脚本化模型后端产生确定性工具调用序列，衡量的是运行时/工具层/"
                     "记忆/自我修正路径的可靠性；换真实模型后同一套任务与指标可复用。")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #

AgentFactory = Callable[[EvalTask], AgentRuntime]


class SuiteRunner:
    """按任务集跑评测，跑完算指标。"""

    def __init__(self, factory: AgentFactory, label: str = "suite") -> None:
        self.factory = factory
        self.label = label

    def run(self, tasks: Iterable[EvalTask]) -> EvalReport:
        started = time.perf_counter()
        outcomes: List[TaskOutcome] = []
        for task in tasks:
            try:
                agent = self.factory(task)
                result = agent.run(task.prompt)
                outcomes.append(TaskOutcome(task=task, result=result, checks=task.check(result)))
            except Exception as exc:  # 单个任务炸掉不能带走整轮评测
                outcomes.append(
                    TaskOutcome(task=task, result=None, checks={}, error=f"{type(exc).__name__}: {exc}")
                )
        return EvalReport(
            outcomes=outcomes,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            label=self.label,
        )


# --------------------------------------------------------------------------- #
# 示例场景：设备运维 Agent
# --------------------------------------------------------------------------- #

METRIC_TABLE: Dict[Tuple[str, str], List[float]] = {
    ("press-01", "temperature"): [61.2, 63.8, 68.4, 74.9, 81.3],
    ("press-01", "vibration"): [1.8, 2.1, 2.4, 3.9, 5.2],
    ("press-01", "current"): [11.2, 11.5, 12.1, 13.8, 15.4],
    ("pump-07", "temperature"): [42.1, 42.6, 43.0, 43.4, 43.9],
    ("pump-07", "vibration"): [0.9, 1.0, 1.0, 1.1, 1.2],
    ("pump-07", "current"): [6.1, 6.2, 6.2, 6.3, 6.4],
    ("cnc-03", "temperature"): [35.5, 40.2, 47.7, 55.1, 62.8],
    ("cnc-03", "vibration"): [2.2, 2.6, 3.1, 4.4, 6.1],
    ("cnc-03", "current"): [8.8, 9.1, 9.6, 10.9, 12.7],
}

KNOWLEDGE_DOCS: Dict[str, str] = {
    "sop-overheat": (
        "设备过热处置规程：当主轴温度在 10 分钟窗口内升幅超过 15 摄氏度，判定为温升异常。"
        "处置顺序为：① 降低主轴转速至额定值的 60%；② 检查冷却液流量是否低于 12 L/min；"
        "③ 若 5 分钟内温度未回落，执行停机并通知设备工程师。"
        "注意：温度绝对值低于 70 摄氏度但温升速率过快时，同样按温升异常处理。"
    ),
    "sop-vibration": (
        "振动异常处置规程：振动速度有效值超过 4.5 mm/s 时判定为异常。"
        "常见原因是轴承磨损与地脚螺栓松动；处置顺序为紧固地脚螺栓、复测振动、"
        "若仍超标则更换轴承。"
    ),
    "sop-report": (
        "巡检报告规范：报告需包含设备编号、观测窗口、关键指标与判定结论四要素，"
        "判定结论必须给出依据（引用规程条款或阈值），不得只给结论。"
    ),
}

TOOL_RUNBOOK: Dict[str, str] = {
    "query_metric": "查询设备在最近一个窗口内的指标时间序列",
    "trend": "对时间序列做最小二乘线性拟合，给出斜率与拟合优度",
    "convert_unit": "在常用工程单位之间换算（温度、速度、压力）",
    "calculator": "计算数学表达式",
    "search_knowledge_base": "检索本地处置规程知识库",
    "write_report": "把巡检结论写入沙箱文件",
    "read_file": "读取沙箱文件",
    "list_files": "列出沙箱目录",
    "search_files": "在沙箱文件中检索关键字",
}


def build_ops_tools(kb: KnowledgeBase, sandbox_root: Any, flaky: bool = True) -> ToolRegistry:
    """构建设备运维场景的工具集（全部零依赖、纯函数、结果确定）。"""
    registry = ToolRegistry([calculator])
    registry.register(build_search_tool(kb))

    devices = tuple(sorted({d for d, _ in METRIC_TABLE}))
    metrics = tuple(sorted({m for _, m in METRIC_TABLE}))

    def query_metric(device: str, metric: str, window_min: int = 10) -> Dict[str, Any]:
        """查询设备指标在最近一个时间窗口内的序列。

        Args:
            device: 设备编号。
            metric: 指标名。
            window_min: 观察窗口（分钟），1~120。
        """
        if not 1 <= int(window_min) <= 120:
            raise ValueError(f"窗口长度必须在 1~120 分钟，收到 {window_min}")
        key = (device, metric)
        if key not in METRIC_TABLE:
            raise ValueError(f"设备 {device} 没有 {metric} 指标")
        series = METRIC_TABLE[key]
        return {
            "device": device,
            "metric": metric,
            "window_min": int(window_min),
            "series": series,
            "delta": round(series[-1] - series[0], 4),
            "unit": {"temperature": "℃", "vibration": "mm/s", "current": "A"}[metric],
        }

    def trend(values: List[float]) -> Dict[str, Any]:
        """对一组数值做最小二乘线性拟合。

        Args:
            values: 至少 2 个数值的时间序列。
        """
        n = len(values)
        if n < 2:
            raise ValueError("至少需要 2 个点才能拟合")
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(values) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, values))
        slope = sxy / sxx if sxx else 0.0
        intercept = my - slope * mx
        ss_tot = sum((y - my) ** 2 for y in values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, values))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        return {
            "slope": round(slope, 6),
            "intercept": round(intercept, 6),
            "r2": round(r2, 6),
            "n": n,
        }

    units = ("celsius", "fahrenheit", "kelvin")
    speed_units = ("mm/s", "cm/s", "m/s")

    def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
        """在常用工程单位之间换算。温度支持 celsius/fahrenheit/kelvin，速度支持 mm/s、cm/s、m/s。

        Args:
            value: 待换算的数值。
            from_unit: 源单位。
            to_unit: 目标单位。
        """
        if from_unit == to_unit:
            return f"{value}"
        if from_unit in units and to_unit in units:
            celsius = {
                "celsius": value,
                "fahrenheit": (value - 32.0) * 5.0 / 9.0,
                "kelvin": value - 273.15,
            }[from_unit]
            out = {
                "celsius": celsius,
                "fahrenheit": celsius * 9.0 / 5.0 + 32.0,
                "kelvin": celsius + 273.15,
            }[to_unit]
            return f"{round(out, 4)} {to_unit}"
        if from_unit in speed_units and to_unit in speed_units:
            base = {"mm/s": 1.0, "cm/s": 10.0, "m/s": 1000.0}[from_unit] * value
            return f"{round(base / {'mm/s': 1.0, 'cm/s': 10.0, 'm/s': 1000.0}[to_unit], 6)} {to_unit}"
        raise ValueError(f"不支持从 {from_unit} 换算到 {to_unit}")

    sandbox = FileSandbox(Path(sandbox_root))

    def write_report(path: str, content: str) -> str:
        """把巡检结论按报告规范写入沙箱文件。

        Args:
            path: 沙箱内相对路径。
            content: 报告正文。
        """
        return sandbox.write(path, content)

    spec_query = registry.register_function(
        query_metric, name="query_metric", tags=("设备", "数据"),
        description=TOOL_RUNBOOK["query_metric"],
        constraints={
            "device": {"enum": list(devices)},
            "metric": {"enum": list(metrics)},
            "window_min": {"minimum": 1, "maximum": 120},
        },
    )
    registry.register_function(trend, name="trend", tags=("分析",),
                               description=TOOL_RUNBOOK["trend"],
                               constraints={"values": {"minItems": 2}})
    registry.register_function(convert_unit, name="convert_unit", tags=("单位",),
                               description=TOOL_RUNBOOK["convert_unit"],
                               constraints={
                                   "from_unit": {"enum": list(units + speed_units)},
                                   "to_unit": {"enum": list(units + speed_units)},
                               })
    registry.register_function(write_report, name="write_report", tags=("工具", "报告"),
                               description=TOOL_RUNBOOK["write_report"])
    for spec in build_file_tools(sandbox):
        registry.register(spec)

    if flaky:
        # 首次调用必然失败的网关（用于验证「工具报错 → 重试」这条自我修正路径）
        state = {"calls": 0}

        def flaky_gateway(device: str) -> str:
            """通过不稳定网关读取设备心跳，偶发超时。

            Args:
                device: 设备编号。
            """
            state["calls"] += 1
            if state["calls"] == 1:
                raise TimeoutError("设备网关超时（第 1 次请求）")
            return f"{device} 网关连通，心跳正常（第 {state['calls']} 次请求）"

        registry.register_function(
            flaky_gateway, name="flaky_gateway", tags=("设备", "网关"),
            description="通过不稳定网关读取设备心跳，可能超时",
        )

    return registry


def build_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase(dim=512, size=200, overlap=40)
    for doc_id, text in KNOWLEDGE_DOCS.items():
        kb.add_document(doc_id, text, metadata={"kind": "sop"})
    return kb


def build_suite_tasks() -> List[EvalTask]:
    """12 个评测任务，覆盖：单工具、多工具串联、RAG、参数纠错、工具报错恢复、
    未知工具纠偏、死循环拦截、纯问答、写文件。"""
    return [
        EvalTask(
            id="calc-thermal-margin",
            prompt="散热余量计算：额定 85 摄氏度，实测 68.4 摄氏度，余量百分比是多少？",
            script=[
                {"tool_calls": [{"name": "calculator", "arguments": {"expression": "(85-68.4)/85*100"}}]},
                "散热余量约为 19.5%。",
            ],
            expected_tools=("calculator",),
            expected_answer_contains=("19.5",),
            expected_sequence=("calculator",),
            tags=("单工具", "计算"),
        ),
        EvalTask(
            id="metric-window-delta",
            prompt="查一下 press-01 最近 10 分钟的温度指标。",
            script=[
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "press-01", "metric": "temperature",
                                               "window_min": 10}}]},
                "press-01 温度序列已获取，末值 81.3 ℃。",
            ],
            expected_tools=("query_metric",),
            expected_answer_contains=("81.3",),
            tags=("单工具", "取数"),
        ),
        EvalTask(
            id="unit-convert",
            prompt="把 81.3 摄氏度换算成华氏度。",
            script=[
                {"tool_calls": [{"name": "convert_unit",
                                 "arguments": {"value": 81.3, "from_unit": "celsius",
                                               "to_unit": "fahrenheit"}}]},
                "约 178.34 华氏度。",
            ],
            expected_tools=("convert_unit",),
            expected_answer_contains=("178.34",),
            tags=("单工具", "单位"),
        ),
        EvalTask(
            id="rag-overheat-sop",
            prompt="主轴温度 10 分钟内升幅过大，应该按什么规程处置？",
            script=[
                {"tool_calls": [{"name": "search_knowledge_base",
                                 "arguments": {"query": "温升异常 处置规程", "k": 3}}]},
                "按 sop-overheat 规程：转速降至 60%，检查冷却液流量，5 分钟未回落则停机。",
            ],
            expected_tools=("search_knowledge_base",),
            expected_answer_contains=("sop-overheat", "60%"),
            tags=("RAG", "检索"),
        ),
        EvalTask(
            id="invalid-arguments-recovery",
            prompt="查 cnc-03 的温度指标，窗口取 10 分钟。",
            script=[
                # 第一次故意把 window_min 写成 0（超范围）→ 被运行时前置拦截
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "cnc-03", "metric": "temperature",
                                               "window_min": 0}}]},
                # 收到字段级错误后修正参数
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "cnc-03", "metric": "temperature",
                                               "window_min": 10}}]},
                "cnc-03 温度末值 62.8 ℃。",
            ],
            expected_tools=("query_metric",),
            expected_answer_contains=("62.8",),
            tags=("自我修正", "参数校验"),
            note="第一次参数越界必须被拦在执行之前，模型看到字段级错误后修正参数",
        ),
        EvalTask(
            id="unknown-tool-recovery",
            prompt="用 query_metric_v2 接口查一下 pump-07 的振动。",
            script=[
                {"tool_calls": [{"name": "query_metric_v2",
                                 "arguments": {"device": "pump-07", "metric": "vibration"}}]},
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "pump-07", "metric": "vibration",
                                               "window_min": 10}}]},
                "pump-07 振动末值 1.2 mm/s，处于正常范围。",
            ],
            expected_tools=("query_metric",),
            expected_answer_contains=("1.2",),
            forbidden_tools=("query_metric_v2",),
            tags=("自我修正", "未知工具"),
            note="工具名不存在时要给出候选建议，而不是直接崩",
        ),
        EvalTask(
            id="tool-error-retry",
            prompt="读取 pump-07 的网关心跳。",
            script=[
                {"tool_calls": [{"name": "flaky_gateway", "arguments": {"device": "pump-07"}}]},
                {"tool_calls": [{"name": "flaky_gateway", "arguments": {"device": "pump-07"}}]},
                "网关已连通，pump-07 心跳正常。",
            ],
            expected_tools=("flaky_gateway",),
            expected_answer_contains=("心跳正常",),
            expect_recovery=True,
            tags=("自我修正", "工具报错"),
        ),
        EvalTask(
            id="loop-guard",
            prompt="反复查询 press-01 的电流，直到我喊停。",
            script=[
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "press-01", "metric": "current",
                                               "window_min": 10}}]},
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "press-01", "metric": "current",
                                               "window_min": 10}}]},
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "press-01", "metric": "current",
                                               "window_min": 10}}]},
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "press-01", "metric": "current",
                                               "window_min": 10}}]},
                "press-01 电流峰值 15.4 A，已停止重复查询。",
            ],
            expected_tools=("query_metric",),
            expected_answer_contains=("15.4",),
            tags=("自我修正", "死循环"),
            note="同一签名的第 3 次调用会被运行时拦截并回填提示，模型据此收敛",
        ),
        EvalTask(
            id="multi-step-analysis",
            prompt="查 cnc-03 的温度并拟合并评价温升趋势，最后给出结论。",
            script=[
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "cnc-03", "metric": "temperature",
                                               "window_min": 10}}]},
                {"tool_calls": [{"name": "trend",
                                 "arguments": {"values": [35.5, 40.2, 47.7, 55.1, 62.8]}}]},
                "cnc-03 温度斜率约 6.95 ℃/min，呈持续上升趋势，判定为温升异常。",
            ],
            expected_tools=("query_metric", "trend"),
            expected_answer_contains=("6.95", "异常"),
            expected_sequence=("query_metric", "trend"),
            tags=("多步", "分析"),
        ),
        EvalTask(
            id="rag-then-decide",
            prompt="69 摄氏度的主轴温度算不算异常？请给出处置建议。",
            script=[
                {"tool_calls": [{"name": "search_knowledge_base",
                                 "arguments": {"query": "温度异常 判定阈值 处置", "k": 3}}]},
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "press-01", "metric": "temperature",
                                               "window_min": 10}}]},
                "绝对值未超 70 摄氏度，但升幅 20.1 ℃ 超过 15 ℃ 门限，按温升异常处置。",
            ],
            expected_tools=("search_knowledge_base", "query_metric"),
            expected_answer_contains=("20.1",),
            tags=("RAG", "多步"),
        ),
        EvalTask(
            id="no-tool-needed",
            prompt="你好，请用一句话说明你能做什么。",
            script=["我可以查询设备指标、检索处置规程并计算工程数值。"],
            expected_tools=(),
            expected_answer_contains=("设备指标",),
            tags=("纯问答",),
        ),
        EvalTask(
            id="plan-and-replan",
            prompt="排查 cnc-03 的温升问题并给出处置建议。",
            script=[
                '{"goal": "排查 cnc-03 温升", "steps": ['
                '{"description": "读取设备网关状态", "tool": "flaky_gateway"}, '
                '{"description": "给出处置结论"}]}',
                # 第 1 步失败 → 运行时应当触发重规划
                {"tool_calls": [{"name": "flaky_gateway", "arguments": {"device": "cnc-03"}}]},
                '{"steps": ["改用 query_metric 直接取温度", "检索处置规程", "给出结论"]}',
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "cnc-03", "metric": "temperature",
                                               "window_min": 10}}]},
                {"tool_calls": [{"name": "search_knowledge_base",
                                 "arguments": {"query": "温升异常 处置规程", "k": 3}}]},
                "cnc-03 温升异常，按 sop-overheat 规程降转速并检查冷却液。",
            ],
            expected_tools=("query_metric", "search_knowledge_base"),
            expected_answer_contains=("sop-overheat",),
            plan_first=True,
            expect_replans=1,
            max_steps=6,
            tags=("规划", "重规划"),
            note="先规划后执行；首步失败后保留已完成步骤、只重写剩余路径",
        ),
        EvalTask(
            id="context-compaction",
            prompt="依次查询 press-01、pump-07、cnc-03 的温度并汇总最大温升。",
            script=[
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "press-01", "metric": "temperature",
                                               "window_min": 10}}]},
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "pump-07", "metric": "temperature",
                                               "window_min": 10}}]},
                {"tool_calls": [{"name": "query_metric",
                                 "arguments": {"device": "cnc-03", "metric": "temperature",
                                               "window_min": 10}}]},
                "三台设备中 press-01 温升最大（20.1 ℃），末值 81.3 ℃。",
            ],
            expected_tools=("query_metric",),
            expected_answer_contains=("81.3", "20.1"),
            context_budget_tokens=260,
            expect_compaction=True,
            tags=("上下文", "记忆"),
            note="预算吃紧时必须压缩历史，且 assistant(tool_calls)/tool 必须成对保留",
        ),
        EvalTask(
            id="write-report-file",
            prompt="把 cnc-03 温升异常的结论写成报告存到 reports/cnc-03.md。",
            script=[
                {"tool_calls": [{"name": "write_report",
                                 "arguments": {"path": "reports/cnc-03.md",
                                               "content": "设备 cnc-03 温升异常：斜率 6.95 ℃/min"}}]},
                "报告已写入 reports/cnc-03.md。",
            ],
            expected_tools=("write_report",),
            expected_answer_contains=("reports/cnc-03.md",),
            tags=("工具", "写文件"),
        ),
    ]


def make_scenario_factory(
    sandbox_root: Any,
    config: Optional[AgentConfig] = None,
    use_memory: bool = True,
) -> AgentFactory:
    """构造「每个任务一个干净运行时」的工厂。

    ⚠️ **工具集与知识库每个任务重建一次，绝不跨任务共享**。
    这是踩过的坑：``flaky_gateway`` 这类工具带内部状态（第一次调用必失败），
    如果整轮评测共用一个注册表，先跑的用例会把「第一次失败」用掉，
    后面依赖这次失败的用例就**静默走了另一条路径**（不报错、只是结果变了）。
    评测用例之间必须完全隔离，否则顺序会影响指标 —— 而顺序相关的指标是不可信的。
    """

    def factory(task: EvalTask) -> AgentRuntime:
        kb = build_knowledge_base()
        registry = build_ops_tools(kb, sandbox_root)
        cfg = config if config is not None else AgentConfig(
            max_steps=task.max_steps,
            plan_first=task.plan_first,
            context_budget_tokens=task.context_budget_tokens or 4000,
        )
        return AgentRuntime(
            ScriptedBackend(task.script),
            registry,
            config=cfg,
            memory=ShortTermMemory(
                max_tokens=cfg.context_budget_tokens if use_memory else 10 ** 9
            ),
            working=WorkingMemory(),
            knowledge=kb,
            name=f"eval:{task.id}",
        )

    return factory


def run_default_suite(sandbox_root: Any, label: Optional[str] = None) -> EvalReport:
    """跑一遍内置评测集。"""
    tasks = build_suite_tasks()
    runner = SuiteRunner(
        make_scenario_factory(sandbox_root),
        label=label or f"设备运维 Agent（{len(tasks)} 任务）",
    )
    return runner.run(tasks)


def scenario_summary(registry: ToolRegistry, kb: KnowledgeBase) -> Dict[str, Any]:
    """场景概览（给 README 与报告用）。"""
    return {
        "tools": registry.names,
        "tool_count": len(registry),
        "knowledge_docs": len(kb.docs),
        "knowledge_chunks": len(kb),
        "tasks": len(build_suite_tasks()),
    }
