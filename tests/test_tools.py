"""工具层测试：schema 生成、参数校验、注册表、沙箱、安全计算器。"""

from __future__ import annotations

import pytest

from agentlab.tools import (
    FileSandbox,
    ToolNotFound,
    ToolRegistry,
    ToolValidationError,
    build_file_tools,
    calculator,
    safe_eval,
    tool,
    validate_arguments,
)


# --------------------------------------------------------------------------- #
# schema 生成
# --------------------------------------------------------------------------- #

def test_schema_from_signature_and_docstring():
    def sample(device: str, window: int = 10, ratio: float = 1.0, flag: bool = False) -> str:
        """采集设备数据。

        Args:
            device: 设备编号。
            window: 观察窗口（分钟）。
        """
        return device

    registry = ToolRegistry()
    spec = registry.register_function(sample, name="sample")
    props = spec.parameters["properties"]
    assert props["device"]["type"] == "string"
    assert props["device"]["description"] == "设备编号。"
    assert props["window"]["type"] == "integer"
    assert props["window"]["description"] == "观察窗口（分钟）。"
    assert props["window"]["default"] == 10
    assert props["ratio"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"
    assert spec.parameters["required"] == ["device"]
    assert spec.description == "采集设备数据。"
    assert spec.signature_hint() == "sample(device:string, window?:integer, ratio?:number, flag?:boolean)"


def test_typed_annotations_survive_postponed_evaluation():
    """回归测试：``from __future__ import annotations`` 不能让 schema 变成空的。

    本模块顶部就写了该 import，所以 ``sample`` 的注解在运行期是字符串。
    如果实现里没有做 ``get_type_hints`` 求值，生成的 schema 会**没有 type 字段**，
    参数校验也就永远不会触发 —— 这是一个「不报错但功能全失效」的坑。
    """
    def local(x: int, items: list) -> None:
        """本地函数（注解同样被延迟求值）。

        Args:
            x: 整数。
            items: 列表。
        """
        return None

    spec = ToolRegistry().register_function(local, name="local")
    assert spec.parameters["properties"]["x"]["type"] == "integer", "注解未被解析，schema 缺失 type"
    assert spec.parameters["properties"]["items"]["type"] == "array"


def test_optional_and_container_annotations():
    from typing import Dict, List, Optional

    def f(a: Optional[int] = None, b: List[float] = (), c: Dict[str, int] = {}) -> None:
        """容器与可选类型。

        Args:
            a: 可选整数。
            b: 浮点列表。
            c: 字典。
        """
        return None

    spec = ToolRegistry().register_function(f, name="f")
    props = spec.parameters["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "array"
    assert props["b"]["items"] == {"type": "number"}
    assert props["b"]["default"] == ()
    assert props["b"]["description"] == "浮点列表。"
    assert props["c"]["type"] == "object"
    assert spec.parameters.get("required", []) == []


def test_constraints_are_merged_into_schema():
    def g(window_min: int = 10, mode: str = "fast") -> str:
        """带约束的工具。

        Args:
            window_min: 窗口。
            mode: 模式。
        """
        return mode

    spec = ToolRegistry().register_function(
        g, name="g",
        constraints={"window_min": {"minimum": 1, "maximum": 120}, "mode": {"enum": ["fast", "slow"]}},
    )
    props = spec.parameters["properties"]
    assert props["window_min"]["type"] == "integer"
    assert (props["window_min"]["minimum"], props["window_min"]["maximum"]) == (1, 120)
    assert props["mode"]["enum"] == ["fast", "slow"]
    # 约束生效后才能被前置拦截
    assert validate_arguments(spec.parameters, {"window_min": 0}) == ["参数 'window_min' 小于最小值 1"]
    assert validate_arguments(spec.parameters, {"mode": "turbo"}) == [
        "参数 'mode' 取值非法: 'turbo' 不在 ['fast', 'slow'] 内"
    ]


def test_tool_decorator_attaches_spec():
    @tool(name="ping", tags=("网络",), description="探活")
    def ping(host: str, times: int = 1) -> str:
        """探活。

        Args:
            host: 主机。
            times: 次数。
        """
        return host * times

    spec = ping.__tool_spec__
    assert spec.name == "ping" and spec.tags == ("网络",) and spec.description == "探活"
    assert ToolRegistry([ping]).names == ["ping"]


def test_to_openai_shape():
    registry = ToolRegistry([calculator])
    wire = registry.to_openai()[0]
    assert wire["type"] == "function"
    assert set(wire["function"]) == {"name", "description", "parameters"}
    assert wire["function"]["name"] == "calculator"


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #

SCHEMA = {
    "type": "object",
    "properties": {
        "device": {"type": "string", "enum": ["a", "b"]},
        "count": {"type": "integer", "minimum": 1, "maximum": 5},
        "values": {"type": "array", "items": {"type": "number"}, "minItems": 2},
        "note": {"type": "string", "maxLength": 4},
    },
    "required": ["device", "count"],
}


@pytest.mark.parametrize(
    "args, expected_fragment",
    [
        ({"count": 2}, "缺少必填参数 'device'"),
        ({"device": None, "count": 2}, "缺少必填参数 'device'"),
        ({"device": "a"}, "缺少必填参数 'count'"),
        ({"device": "a", "count": 2, "extra": 1}, "未知参数 'extra'"),
        ({"device": "c", "count": 2}, "取值非法"),
        ({"device": "a", "count": "2"}, "类型错误"),
        ({"device": "a", "count": 9}, "大于最大值 5"),
        ({"device": "a", "count": 0}, "小于最小值 1"),
        ({"device": "a", "count": 2, "values": ["x", "y"]}, "参数 'values[0]' 类型错误"),
        ({"device": "a", "count": 2, "values": [1]}, "元素个数少于 2"),
        ({"device": "a", "count": 2, "note": "12345"}, "超过最大长度 4"),
    ],
)
def test_validate_arguments_reports_field_level_errors(args, expected_fragment):
    errors = validate_arguments(SCHEMA, args)
    assert any(expected_fragment in e for e in errors), errors


def test_validate_arguments_accepts_valid_payload():
    assert validate_arguments(SCHEMA, {"device": "a", "count": 2}) == []
    assert validate_arguments(SCHEMA, {"device": "b", "count": 5, "values": [1, 2.5], "note": "ok"}) == []


def test_validate_arguments_rejects_non_dict():
    assert validate_arguments(SCHEMA, ["not", "a", "dict"]) == ["arguments 必须是对象，收到 list"]


def test_bool_is_not_integer():
    """Python 里 bool 是 int 的子类，但 ``True`` 不该被当成合法的 integer 参数。"""
    errors = validate_arguments({"type": "object", "properties": {"n": {"type": "integer"}}}, {"n": True})
    assert errors and "类型错误" in errors[0]


# --------------------------------------------------------------------------- #
# 注册表与调用
# --------------------------------------------------------------------------- #

def test_registry_lookup_and_suggestions():
    registry = ToolRegistry([calculator])
    assert "calculator" in registry and len(registry) == 1
    with pytest.raises(ToolNotFound) as exc:
        registry.get("calculator_v2")
    assert "calculator_v2" in str(exc.value)
    assert "calculator" in str(exc.value)  # difflib 给出了候选
    assert registry.get("calculator").name == "calculator"


def test_call_does_not_execute_when_arguments_invalid():
    executed = []

    def spy(x: int) -> int:
        """会被记一次。

        Args:
            x: 参数。
        """
        executed.append(x)
        return x

    registry = ToolRegistry()
    registry.register_function(spy, name="spy", constraints={"x": {"minimum": 1}})
    with pytest.raises(ToolValidationError) as exc:
        registry.call("spy", {"x": 0})
    assert executed == [], "参数非法时绝不能执行工具函数"
    assert "小于最小值 1" in str(exc.value)
    assert exc.value.errors == ["参数 'x' 小于最小值 1"]


def test_call_wraps_tool_exceptions():
    def boom() -> None:
        """总是失败。"""
        raise TimeoutError("网关超时")

    registry = ToolRegistry()
    registry.register_function(boom, name="boom")
    result = registry.call("boom", {})
    assert result.ok is False
    assert "网关超时" in result.error and "TimeoutError" in result.error
    assert result.to_observation().startswith("ERROR:")


def test_call_reports_unknown_tool_as_failed_result():
    registry = ToolRegistry([calculator])
    result = registry.call("nope", {})
    assert result.ok is False and "未注册的工具" in result.error


def test_call_retries_internal_failures():
    state = {"n": 0}

    def flaky() -> str:
        """前两次失败。"""
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("瞬时错误")
        return "ok"

    registry = ToolRegistry()
    registry.register_function(flaky, name="flaky")
    assert registry.call("flaky", {}, max_retries=0).ok is False
    state["n"] = 0
    result = registry.call("flaky", {}, max_retries=2)
    assert result.ok is True and result.attempts == 3


def test_observation_truncates_long_output():
    from agentlab.tools import ToolResult

    result = ToolResult(name="x", ok=True, value="a" * 5000)
    obs = result.to_observation(max_chars=100)
    assert len(obs) < 200 and "截断" in obs


def test_by_tag_and_catalog():
    registry = ToolRegistry([calculator])
    assert [s.name for s in registry.by_tag("计算")] == ["calculator"]
    assert "calculator(expression:string)" in registry.catalog()


# --------------------------------------------------------------------------- #
# 安全计算器
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "expr, expected",
    [
        ("(12+8)*3/7", "8.571428571428571"),
        ("sqrt(2)*100", "141.4213562373095"),
        ("2**10", "1024"),
        ("abs(-3)+max(1,2,3)", "6"),
        ("pi", "3.141592653589793"),
        ("7//2", "3"),
        ("10 % 3", "1"),
    ],
)
def test_calculator_computes(expr, expected):
    assert calculator(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "(1).__class__",
        "open('/etc/passwd')",
        "lambda: 1",
        "[x for x in range(3)]",
        "2**999",
        "1/0",
        "print(1)",
    ],
)
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises((ValueError, ZeroDivisionError, SyntaxError, TypeError)):
        safe_eval(expr)


def test_calculator_returns_integer_like_floats():
    assert calculator("85 - 68.4") == "16.6"
    assert calculator("10/2") == "5"


# --------------------------------------------------------------------------- #
# 文件沙箱
# --------------------------------------------------------------------------- #

def test_sandbox_blocks_escape(tmp_path):
    box = FileSandbox(tmp_path / "root")
    with pytest.raises(ValueError):
        box.resolve("../outside.txt")
    with pytest.raises(ValueError):
        box.read("../../secret.txt")


def test_sandbox_roundtrip_and_search(tmp_path):
    box = FileSandbox(tmp_path / "root")
    assert "已写入" in box.write("a/b.txt", "hello world\nsecond line")
    assert box.read("a/b.txt").startswith("hello")
    assert sorted(box.list_dir(".")) == ["a/"]
    assert sorted(box.list_dir("a")) == ["b.txt"]
    hits = box.search("second")
    assert hits and hits[0]["file"] == "a/b.txt" and hits[0]["line"] == 2


def test_sandbox_reads_only_utf8_text(tmp_path):
    box = FileSandbox(tmp_path / "root")
    (box.root / "bin.dat").write_bytes(b"\xff\xfe\x00\x01")
    assert box.search("x") == []  # 二进制文件被跳过而不是抛异常


def test_file_tools_expose_schema(tmp_path):
    specs = build_file_tools(FileSandbox(tmp_path))
    names = {s.name for s in specs}
    assert names == {"read_file", "write_file", "list_files", "search_files"}
    write_spec = next(s for s in specs if s.name == "write_file")
    assert write_spec.parameters["required"] == ["path", "content"]
    assert write_spec.parameters["properties"]["content"]["type"] == "string"
