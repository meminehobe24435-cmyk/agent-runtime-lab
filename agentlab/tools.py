"""工具层：JSON Schema 生成、参数校验、Function Calling 协议编解码。

Agent 的「工具调用」能不能落地，取决于三件事：

1. **模型能否看懂有哪些工具** → :func:`tool` 装饰器从函数签名与 docstring 自动生成
   JSON Schema，再转成 OpenAI function calling 的 ``tools`` 数组；
2. **模型给的参数能不能信** → :func:`validate_arguments` 在**执行前**逐项校验
   （类型 / 必填 / 枚举 / 数值范围 / 数组元素），不合法就抛
   :class:`ToolValidationError`，由运行时交回模型自我修正；
3. **工具失败会不会把 Agent 带崩** → 所有工具统一返回 :class:`ToolResult`，
   异常被捕获成结构化错误，而不是终止整个任务。
"""

from __future__ import annotations

import ast
import difflib
import inspect
import json
import math
import operator
import time
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, get_args, get_origin


class ToolError(Exception):
    """工具层错误基类。"""


class ToolNotFound(ToolError):
    """调用了未注册的工具。"""

    def __init__(self, name: str, candidates: Sequence[str] = ()) -> None:
        self.name = name
        self.candidates = list(candidates)
        hint = ""
        close = difflib.get_close_matches(name, list(candidates), n=3, cutoff=0.4)
        if close:
            hint = f"，是否想调用: {', '.join(close)}"
        super().__init__(f"未注册的工具 '{name}'{hint}")


class ToolValidationError(ToolError):
    """参数不符合 schema。``errors`` 是可直接回填给模型的字段级错误列表。"""

    def __init__(self, name: str, errors: Sequence[str]) -> None:
        self.name = name
        self.errors = list(errors)
        super().__init__(f"工具 '{name}' 参数校验失败: " + "; ".join(self.errors))


# --------------------------------------------------------------------------- #
# 类型 → JSON Schema
# --------------------------------------------------------------------------- #

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    dict: "object",
    type(None): "null",
}


def _json_type(annotation: Any) -> Dict[str, Any]:
    """把 Python 类型注解翻译成 JSON Schema 片段。"""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        # Optional[X] / Union[X, None]
        if origin is type(Optional[int]) or str(origin) == "typing.Union":
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return _json_type(non_none[0])
            return {"anyOf": [_json_type(a) for a in non_none]}
        if origin in (list, tuple, set, frozenset):
            item = _json_type(args[0]) if args else {}
            return {"type": "array", "items": item} if item else {"type": "array"}
        if origin is dict:
            return {"type": "object"}
    if isinstance(annotation, type) and annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}
    # 枚举类
    if isinstance(annotation, type) and issubclass(annotation, str):
        return {"type": "string"}
    return {}


def _resolved_hints(func: Callable[..., Any]) -> Dict[str, Any]:
    """把函数注解**真正解析出来**，而不是拿到一串字符串。

    ⚠️ 这是一个踩过的坑：本文件（以及任何写了 ``from __future__ import annotations``
    的模块）里的注解在运行期是 **字符串**（PEP 563），
    ``inspect.signature()`` 拿到的是 ``'int'`` 而不是 ``int``。
    如果直接把它喂给类型映射，会**静默生成一个没有 ``type`` 的空 schema** ——
    校验永远通过、错误参数的模型也永远不被拦下，而且不报任何错。

    解析分两级：先让 ``get_type_hints`` 按标准规则求值；
    失败时（例如注解引用了函数内局部导入的名字）退化为**逐参数**在
    「模块全局 + typing 命名空间」里求值；仍失败就退化为「不约束」，绝不抛异常。
    """
    try:
        return dict(typing.get_type_hints(func))
    except Exception:
        pass

    namespace: Dict[str, Any] = dict(vars(typing))
    namespace.update(getattr(func, "__globals__", {}) or {})
    resolved: Dict[str, Any] = {}
    for pname, param in inspect.signature(func).parameters.items():
        ann = param.annotation
        if isinstance(ann, str):
            try:
                resolved[pname] = eval(ann, namespace)  # noqa: S307 - 注解来自本仓库源码
            except Exception:
                continue
        elif ann is not inspect.Parameter.empty:
            resolved[pname] = ann
    return resolved


def _parse_docstring_args(doc: Optional[str]) -> Dict[str, str]:
    """从 docstring 的 Args 段落里抽取「参数名 → 说明」。

    只认最朴素的一种写法（每行 ``name: 说明``，可续行），
    够用且不会把 Returns / Raises 段误读成参数。
    """
    if not doc:
        return {}
    out: Dict[str, str] = {}
    lines = inspect.cleandoc(doc).splitlines()
    in_args = False
    current: Optional[str] = None
    for raw in lines:
        stripped = raw.strip()
        if not in_args:
            if stripped in ("Args:", "Arguments:", "参数:", "参数："):
                in_args = True
            continue
        if not stripped:
            continue
        # 顶格的 "Returns:" / "Raises:" 表示 Args 段已经结束
        if raw == stripped and stripped.endswith((":", "：")):
            break
        head, sep, tail = stripped.partition(":")
        if sep and head and " " not in head:
            current = head
            out[current] = tail.strip()
        elif current:
            out[current] = (out[current] + " " + stripped).strip()
    return out


def _summary(doc: Optional[str]) -> str:
    if not doc:
        return ""
    lines = [l.strip() for l in inspect.cleandoc(doc).splitlines()]
    parts: List[str] = []
    for line in lines:
        if not line:
            break
        if line.endswith((":", "：")) and len(line) < 12:
            break
        parts.append(line)
    return " ".join(parts)


@dataclass
class ToolSpec:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]
    tags: Tuple[str, ...] = ()
    timeout_s: Optional[float] = None
    dangerous: bool = False

    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def required(self) -> List[str]:
        return list(self.parameters.get("required") or [])

    def signature_hint(self) -> str:
        """给模型看的紧凑签名，用于参数出错时回填提示。"""
        props = self.parameters.get("properties") or {}
        bits = []
        for key, spec in props.items():
            t = spec.get("type", "any")
            mark = "" if key in self.required else "?"
            enum = spec.get("enum")
            if enum:
                t = "|".join(str(e) for e in enum)
            bits.append(f"{key}{mark}:{t}")
        return f"{self.name}({', '.join(bits)})"


def tool(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Sequence[str] = (),
    timeout_s: Optional[float] = None,
    dangerous: bool = False,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Any:
    """把一个普通函数注册成工具，schema 从签名与 docstring 自动生成。

    ``constraints`` 用来补签名表达不了的约束（取值范围 / 枚举 / 长度上限），
    例如 ``{"window_min": {"minimum": 1, "maximum": 120}}`` ——
    这些约束会被 :func:`validate_arguments` 在**执行前**检查。
    """

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "__tool_spec__", _build_spec(
            func, name=name, description=description, tags=tags,
            timeout_s=timeout_s, dangerous=dangerous, constraints=constraints,
        ))
        return func

    if _func is not None:
        return wrap(_func)
    return wrap


def _build_spec(
    func: Callable[..., Any],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Sequence[str] = (),
    timeout_s: Optional[float] = None,
    dangerous: bool = False,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ToolSpec:
    sig = inspect.signature(func)
    hints = _resolved_hints(func)
    docs = _parse_docstring_args(func.__doc__)
    extra = constraints or {}
    props: Dict[str, Any] = {}
    required: List[str] = []
    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema = _json_type(hints.get(pname, param.annotation))
        schema.update(extra.get(pname) or {})
        if pname in docs:
            schema["description"] = docs[pname]
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        elif param.default is not None and "default" not in schema:
            # 把默认值写进 schema，模型更容易给对参数
            schema["default"] = param.default
        props[pname] = schema
    parameters: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        parameters["required"] = required
    return ToolSpec(
        name=name or func.__name__,
        description=description or _summary(func.__doc__) or (name or func.__name__),
        parameters=parameters,
        func=func,
        tags=tuple(tags),
        timeout_s=timeout_s,
        dangerous=dangerous,
    )


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #

def _check_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def validate_arguments(
    schema: Dict[str, Any], arguments: Dict[str, Any]
) -> List[str]:
    """校验参数字典，返回字段级错误列表（空列表 = 通过）。

    刻意返回**错误列表**而不是抛异常：运行时需要把这份列表原样交回模型，
    让它知道「哪个字段错、错在哪、正确签名长什么样」。
    """
    errors: List[str] = []
    if not isinstance(arguments, dict):
        return [f"arguments 必须是对象，收到 {type(arguments).__name__}"]

    props: Dict[str, Any] = schema.get("properties") or {}
    required: Sequence[str] = schema.get("required") or []

    for key in required:
        if key not in arguments or arguments[key] is None:
            errors.append(f"缺少必填参数 '{key}'")

    extra = [k for k in arguments if k not in props]
    for key in extra:
        errors.append(f"未知参数 '{key}'（可用: {', '.join(props) or '无'}）")

    for key, value in arguments.items():
        spec = props.get(key)
        if not spec or value is None:
            continue
        expected = spec.get("type")
        if expected and not _check_type(value, expected):
            errors.append(
                f"参数 '{key}' 类型错误: 需要 {expected}，收到 "
                f"{type(value).__name__}({value!r})"
            )
            continue
        enum = spec.get("enum")
        if enum and value not in enum:
            errors.append(f"参数 '{key}' 取值非法: {value!r} 不在 {list(enum)} 内")
        if expected in ("integer", "number"):
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"参数 '{key}' 小于最小值 {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(f"参数 '{key}' 大于最大值 {spec['maximum']}")
        if expected == "string":
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                errors.append(f"参数 '{key}' 超过最大长度 {spec['maxLength']}")
        if expected == "array":
            if "minItems" in spec and len(value) < spec["minItems"]:
                errors.append(f"参数 '{key}' 元素个数少于 {spec['minItems']}")
            if "maxItems" in spec and len(value) > spec["maxItems"]:
                errors.append(f"参数 '{key}' 元素个数多于 {spec['maxItems']}")
            items = spec.get("items") or {}
            item_type = items.get("type")
            if item_type:
                for idx, item in enumerate(value):
                    if not _check_type(item, item_type):
                        errors.append(
                            f"参数 '{key}[{idx}]' 类型错误: 需要 {item_type}"
                        )
    return errors


# --------------------------------------------------------------------------- #
# 结果与注册表
# --------------------------------------------------------------------------- #

@dataclass
class ToolResult:
    """工具执行结果。失败也被结构化，便于回填给模型自我修正。"""

    name: str
    ok: bool
    value: Any = None
    error: str = ""
    elapsed_ms: float = 0.0
    attempts: int = 1

    def to_observation(self, max_chars: int = 1200) -> str:
        if self.ok:
            text = (
                self.value
                if isinstance(self.value, str)
                else json.dumps(self.value, ensure_ascii=False, default=str)
            )
        else:
            text = f"ERROR: {self.error}"
        if len(text) > max_chars:
            text = text[: max_chars - 20] + f"...(截断, 共 {len(text)} 字符)"
        return text


class ToolRegistry:
    """工具注册表。负责注册、查找、协议导出与统一调用。"""

    def __init__(self, tools: Sequence[Any] = ()) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        for item in tools:
            self.register(item)

    # -- 注册 ------------------------------------------------------------- #
    def register(self, item: Any) -> ToolSpec:
        if isinstance(item, ToolSpec):
            spec = item
        elif callable(item):
            spec = getattr(item, "__tool_spec__", None) or _build_spec(item)
        else:
            raise TypeError(f"无法注册为工具: {type(item)!r}")
        self._tools[spec.name] = spec
        return spec

    def register_function(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Sequence[str] = (),
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ToolSpec:
        spec = _build_spec(
            func, name=name, description=description, tags=tags, constraints=constraints
        )
        self._tools[spec.name] = spec
        return spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # -- 查询 ------------------------------------------------------------- #
    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(name, self.names) from None

    def by_tag(self, tag: str) -> List[ToolSpec]:
        return [s for s in self._tools.values() if tag in s.tags]

    def to_openai(self) -> List[Dict[str, Any]]:
        return [s.to_openai() for s in self._tools.values()]

    def catalog(self) -> str:
        """给系统提示词用的紧凑工具清单。"""
        return "\n".join(f"- {s.signature_hint()}: {s.description}" for s in self._tools.values())

    # -- 调用 ------------------------------------------------------------- #
    def call(self, name: str, arguments: Dict[str, Any], max_retries: int = 0) -> ToolResult:
        started = time.perf_counter()
        attempts = 0
        try:
            spec = self.get(name)
        except ToolNotFound as exc:
            return ToolResult(name=name, ok=False, error=str(exc))

        errors = validate_arguments(spec.parameters, arguments)
        if errors:
            raise ToolValidationError(name, errors)

        last_error: Optional[str] = None
        while attempts <= max_retries:
            attempts += 1
            try:
                value = spec.func(**arguments)
                return ToolResult(
                    name=name,
                    ok=True,
                    value=value,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    attempts=attempts,
                )
            except Exception as exc:  # 工具内部的任何异常都不许击穿运行时
                last_error = f"{type(exc).__name__}: {exc}"
        return ToolResult(
            name=name,
            ok=False,
            error=last_error or "未知错误",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            attempts=attempts,
        )


# --------------------------------------------------------------------------- #
# 内置工具：安全计算器 + 沙箱文件工具
# --------------------------------------------------------------------------- #

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "int": int, "float": float,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pow": pow, "ceil": math.ceil, "floor": math.floor,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}


def safe_eval(expression: str) -> float:
    """用 AST 白名单求值，**不使用 eval()**，杜绝 ``__import__`` 之类逃逸。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误: {exc.msg}") from None

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError(f"不支持的常量: {node.value!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("幂次过大，已拒绝执行")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("只允许白名单内的函数调用")
            if node.keywords:
                raise ValueError("不支持关键字参数")
            return _ALLOWED_FUNCS[node.func.id](*[_eval(a) for a in node.args])
        if isinstance(node, (ast.List, ast.Tuple)):
            return [_eval(e) for e in node.elts]
        raise ValueError(f"不支持的语法节点: {type(node).__name__}")

    return _eval(tree)


def _format_number(value: Any) -> str:
    """把计算结果格式化成**给人（和模型）看**的字符串。

    浮点运算会产生 ``16.599999999999994`` 这种二进制噪声，
    模型读到这种观察结果容易被带偏，所以这里做一次「相对误差 1e-12 内优先输出短形式」
    的清洗：既消掉噪声，又保留 ``8.571428571428571`` 这种真实的循环小数位数。
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    if not math.isfinite(value):
        return "结果不是有限数"
    # 容差取 1e-14 相对量级：刚好覆盖浮点二进制噪声（几个 ulp），
    # 又不会把 141.4213562373095 这种真实有效位「洗掉」。
    tolerance = 1e-14 * max(1.0, abs(value))
    if abs(value - round(value)) <= tolerance:
        return str(int(round(value)))
    short = round(value, 10)
    if abs(value - short) <= tolerance:
        return repr(short)
    return repr(value)


@tool(tags=("计算", "math"))
def calculator(expression: str) -> str:
    """计算一个数学表达式，支持 + - * / // % ** 与 sqrt/log/sin 等白名单函数。

    Args:
        expression: 形如 ``(12+8)*3/7`` 或 ``sqrt(2)*100`` 的表达式字符串。
    """
    return _format_number(safe_eval(expression))


class FileSandbox:
    """把文件工具限制在一个根目录内，阻止 ``../`` 越权访问。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"路径越界，被沙箱拒绝: {relative}")
        return candidate

    def read(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {relative}")
        return path.read_text(encoding="utf-8")

    def write(self, relative: str, content: str) -> str:
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"已写入 {relative}（{len(content)} 字符）"

    def list_dir(self, relative: str = ".") -> List[str]:
        path = self.resolve(relative)
        if not path.is_dir():
            raise NotADirectoryError(f"不是目录: {relative}")
        return sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())

    def search(self, keyword: str, relative: str = ".") -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        base = self.resolve(relative)
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            try:
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if keyword in line:
                        hits.append(
                            {
                                "file": str(path.relative_to(self.root)).replace("\\", "/"),
                                "line": lineno,
                                "text": line.strip()[:160],
                            }
                        )
            except (UnicodeDecodeError, OSError):
                continue
        return hits


def build_file_tools(sandbox: FileSandbox) -> List[ToolSpec]:
    """基于沙箱构造一组文件工具（闭包捕获 sandbox，不暴露真实绝对路径）。"""

    def read_file(path: str) -> str:
        """读取沙箱内的文本文件。

        Args:
            path: 相对沙箱根目录的文件路径。
        """
        return sandbox.read(path)

    def write_file(path: str, content: str) -> str:
        """向沙箱内写入文本文件（覆盖写）。

        Args:
            path: 相对沙箱根目录的文件路径。
            content: 要写入的完整内容。
        """
        return sandbox.write(path, content)

    def list_files(path: str = ".") -> List[str]:
        """列出沙箱内某个目录的文件。

        Args:
            path: 相对沙箱根目录的目录路径。
        """
        return sandbox.list_dir(path)

    def search_files(keyword: str, path: str = ".") -> List[Dict[str, Any]]:
        """在沙箱内的文本文件中按关键字检索。

        Args:
            keyword: 要检索的关键字。
            path: 检索起始目录。
        """
        return sandbox.search(keyword, path)

    def make(spec_func: Callable[..., Any], name: str, tags: Sequence[str]) -> ToolSpec:
        return _build_spec(spec_func, name=name, tags=tags)

    return [
        make(read_file, "read_file", ("文件", "sandbox")),
        make(write_file, "write_file", ("文件", "sandbox")),
        make(list_files, "list_files", ("文件", "sandbox")),
        make(search_files, "search_files", ("文件", "sandbox")),
    ]


def default_registry(sandbox_root: Optional[Path] = None) -> ToolRegistry:
    """默认注册表：计算器 +（传了根目录时）一组沙箱文件工具。"""
    registry = ToolRegistry([calculator])
    if sandbox_root is not None:
        for spec in build_file_tools(FileSandbox(sandbox_root)):
            registry.register(spec)
    return registry
