"""记忆管理：短期缓冲、长期向量记忆、工作记忆。

JD 里「记忆管理」通常指的是三件不同的事，本模块把它们拆成三个独立组件：

- :class:`ShortTermMemory`  对话缓冲。带 **token 预算**，超预算时先做摘要压缩、
  再按时间淘汰；模拟的是真实 Agent 里「上下文窗口有限」这个硬约束。
- :class:`VectorMemory`     长期记忆。用 **hashing trick** 把 n-gram 映射到固定维度
  再做 L2 归一化，检索就是一次点积；不引入任何第三方向量库，可持久化到 JSON。
- :class:`WorkingMemory`    工作记忆（草稿纸）。存当前任务的关键事实，**带来源溯源**
  （这条事实是第几步、由哪个工具得到的），避免模型把幻觉当成事实。

这三者共同支撑运行时里的「可靠性 / 可控性」：上下文不溢出、历史可检索、事实可溯源。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .llm import estimate_tokens

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-.]*|\d+(?:\.\d+)?|[\u3400-\u9fff]")


def tokenize(text: str) -> List[str]:
    """把文本切成检索用的 token。

    英文/数字按词切，中文**逐字切开**再在 :func:`embed` 里合成 bi-gram ——
    这样不依赖任何分词库，也能让中文短语的相似度有意义。
    """
    return _TOKEN_RE.findall(text or "")


def _ngrams(tokens: Sequence[str], n: int = 2) -> List[str]:
    if n <= 1 or len(tokens) < n:
        return list(tokens)
    return ["".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _hash_bucket(token: str, dim: int) -> Tuple[int, float]:
    """FNV-1a 哈希 → (桶下标, 符号)。带符号是为了让碰撞相互抵消而不是单边累加。"""
    h = 0x811C9DC5
    for ch in token.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % dim, 1.0 if (h >> 31) & 1 == 0 else -1.0


def embed(text: str, dim: int = 512, unigram_weight: float = 0.3) -> List[float]:
    """把文本编码成 ``dim`` 维 L2 归一化向量（hashing trick + 字符 bi-gram）。

    **为什么单字权重压到 0.3**：中文没有分词器时，单字（温/度/异/常）噪音很大 ——
    「温度异常」和「振动异常」会共用「异」「常」两个单字，
    而真正区分它们的「温度」「振动」是 bi-gram。
    实测中只靠单字时短查询会被共用的高频字带偏、排到错的文档前面；
    把 bi-gram 作为主特征（权重 1.0）、单字降为辅助（0.3）后排序恢复正确。

    **为什么默认 dim 是 512 而不是 256**：哈希碰撞会带来符号相反的串扰。
    实测（4 条查询 × 3 篇文档）：``dim=256`` 时有查询把**无关文档排到了第一位**、
    甚至出现负相似度；``dim>=512`` 时 4/4 排序全部正确且稳定。
    内存换正确性，值得。
    """
    vec = [0.0] * dim
    tokens = tokenize(text)
    for token in tokens:
        idx, sign = _hash_bucket(token, dim)
        vec[idx] += sign * unigram_weight
    for gram in _ngrams(tokens, 2):
        idx, sign = _hash_bucket(gram, dim)
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """两个向量夹角的余弦；因为 :func:`embed` 已归一化，等价于点积。"""
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# 短期记忆
# --------------------------------------------------------------------------- #

@dataclass
class MemoryTurn:
    role: str
    content: str
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = estimate_tokens(self.content)


class ShortTermMemory:
    """带 token 预算的对话缓冲。

    策略：**先摘要压缩、再按时间淘汰**。超预算时把最旧的一批 turn 折叠进
    ``summary``，保留最近 ``keep_recent`` 条对话。

    两个刻意的设计决定：

    1. **``add()`` 自己就会压缩**，不需要调用方记得调用 :meth:`enforce_budget`
       —— 否则「忘了压缩」就等于静默丢历史；
    2. **没有 summarizer 时不是直接丢**，而是把原文截断成 digest 存进 summary，
       信息降级但不消失；传了 summarizer（例如让模型自己总结）时优先用它。
       summarizer 抛异常也不会影响主流程，自动退回 digest。
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        keep_recent: int = 4,
        summarizer: Optional[Any] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.summarizer = summarizer
        self.turns: List[MemoryTurn] = []
        self.summary: str = ""
        self.compactions = 0
        self.dropped = 0

    # -- 写入 ------------------------------------------------------------- #
    def add(self, role: str, content: str) -> MemoryTurn:
        turn = MemoryTurn(role=role, content=content)
        self.turns.append(turn)
        self.enforce_budget()
        return turn

    def extend(self, items: Iterable[Tuple[str, str]]) -> None:
        for role, content in items:
            self.turns.append(MemoryTurn(role=role, content=content))
        self.enforce_budget()

    # -- 预算 ------------------------------------------------------------- #
    @property
    def tokens(self) -> int:
        return estimate_tokens(self.summary) + sum(t.tokens for t in self.turns)

    @property
    def summary_token_budget(self) -> int:
        """摘要自身也要有上限，否则压缩只是把膨胀从 turns 搬到 summary。"""
        return max(64, self.max_tokens // 2)

    def _digest(self, text: str, summarizer: Optional[Any]) -> str:
        if summarizer is not None:
            try:
                condensed = summarizer(text)
                if condensed:
                    return str(condensed)
            except Exception:  # 摘要失败不能影响主流程
                pass
        return text[:400]

    def enforce_budget(self, summarizer: Optional[Any] = None) -> int:
        """把占用压到预算内，返回本次丢弃的 turn 数。"""
        use = summarizer if summarizer is not None else self.summarizer
        freed = 0
        while self.tokens > self.max_tokens and len(self.turns) > self.keep_recent:
            batch = self.turns[: max(1, len(self.turns) - self.keep_recent)]
            text = "\n".join(f"{t.role}: {t.content}" for t in batch)
            condensed = self._digest(text, use)
            self.summary = (self.summary + "\n" + condensed).strip() if self.summary else condensed
            # 只保留最近的摘要内容（token 预算按字符近似，CJK 场景足够）
            limit = self.summary_token_budget
            if len(self.summary) > limit:
                self.summary = self.summary[-limit:]
            self.compactions += 1
            del self.turns[: len(batch)]
            self.dropped += len(batch)
            freed += len(batch)
        return freed

    def messages(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if self.summary:
            out.append(("system", f"[历史摘要] {self.summary}"))
        out.extend((t.role, t.content) for t in self.turns)
        return out

    def stats(self) -> Dict[str, Any]:
        return {
            "turns": len(self.turns),
            "tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "compactions": self.compactions,
            "dropped": self.dropped,
            "summary_chars": len(self.summary),
        }


# --------------------------------------------------------------------------- #
# 长期向量记忆
# --------------------------------------------------------------------------- #

@dataclass
class MemoryItem:
    item_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: List[float] = field(default_factory=list)
    score: float = 0.0


class VectorMemory:
    """可持久化的向量记忆库（长期记忆）。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._items: Dict[str, MemoryItem] = {}
        self._seq = 0
        self.hits = 0
        self.misses = 0
        self.added = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> List[MemoryItem]:
        return list(self._items.values())

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> MemoryItem:
        self._seq += 1
        key = item_id or f"mem_{self._seq}"
        item = MemoryItem(
            item_id=key,
            text=text,
            metadata=dict(metadata or {}),
            vector=embed(text, self.dim),
        )
        self._items[key] = item
        self.added += 1
        return item

    def extend(self, texts: Iterable[str], metadata: Optional[Dict[str, Any]] = None) -> List[MemoryItem]:
        return [self.add(t, metadata) for t in texts]

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def remove(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    def search(
        self, query: str, top_k: int = 3, min_score: float = 0.0
    ) -> List[MemoryItem]:
        """按余弦相似度检索；返回的条目带 ``score`` 字段（副本，不污染库内数据）。"""
        if not self._items:
            self.misses += 1
            return []
        qvec = embed(query, self.dim)
        scored: List[MemoryItem] = []
        for item in self._items.values():
            score = cosine(qvec, item.vector)
            if score > min_score:
                scored.append(
                    MemoryItem(
                        item_id=item.item_id,
                        text=item.text,
                        metadata=dict(item.metadata),
                        vector=item.vector,
                        score=score,
                    )
                )
        scored.sort(key=lambda it: (-it.score, it.item_id))
        if scored:
            self.hits += 1
        else:
            self.misses += 1
        return scored[:top_k]

    def stats(self) -> Dict[str, Any]:
        return {
            "items": len(self._items),
            "dim": self.dim,
            "hits": self.hits,
            "misses": self.misses,
        }

    # -- 持久化 ----------------------------------------------------------- #
    def save(self, path: Any) -> None:
        payload = {
            "dim": self.dim,
            "seq": self._seq,
            "items": [
                {"id": i.item_id, "text": i.text, "metadata": i.metadata}
                for i in self._items.values()
            ],
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Any) -> "VectorMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        memory = cls(dim=int(data.get("dim") or 512))
        for raw in data.get("items") or []:
            memory.add(raw["text"], raw.get("metadata"), raw.get("id"))
        # 恢复自增序号：既尊重文件里记录的 seq，也不小于已被占用的最大编号，
        # 否则「载入后再新增」会生成与既有条目冲突的 ID。
        saved = int(data.get("seq") or 0)
        used = 0
        for item_id in memory._items:
            if item_id.startswith("mem_") and item_id[4:].isdigit():
                used = max(used, int(item_id[4:]))
        memory._seq = max(saved, used)
        return memory


# --------------------------------------------------------------------------- #
# 工作记忆
# --------------------------------------------------------------------------- #

@dataclass
class Fact:
    key: str
    value: str
    source: str = "unknown"
    step: int = 0


class WorkingMemory:
    """当前任务的草稿纸：键值事实 + 溯源。

    ``source`` 记录这条事实**从哪来**（``tool:query_metric`` / ``user`` / ``rag:doc#3``），
    运行时会把它一并写进提示词，让模型在引用数字时能对上出处，
    也让「模型自己编了一个数」这种情况在 trace 里可查。
    """

    def __init__(self) -> None:
        self._facts: Dict[str, Fact] = {}
        self.conflicts = 0

    def __len__(self) -> int:
        return len(self._facts)

    def set(
        self,
        key: str,
        value: Any,
        source: str = "unknown",
        step: int = 0,
        max_chars: int = 500,
    ) -> Fact:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if len(text) > max_chars:
            # 工作记忆会进提示词，必须限幅：否则一个结构化的大返回值就把上下文吃光
            text = text[:max_chars] + f"…(截断, 共 {len(text)} 字符)"
        existing = self._facts.get(key)
        if existing is not None and existing.value != text:
            self.conflicts += 1
        fact = Fact(key=key, value=text, source=source, step=step)
        self._facts[key] = fact
        return fact

    def get(self, key: str) -> Optional[str]:
        fact = self._facts.get(key)
        return fact.value if fact else None

    def provenance(self, key: str) -> Optional[Fact]:
        return self._facts.get(key)

    def keys(self) -> List[str]:
        return sorted(self._facts)

    def snapshot(self) -> Dict[str, str]:
        return {k: f.value for k, f in self._facts.items()}

    def render(self) -> str:
        if not self._facts:
            return ""
        lines = [
            f"- {f.key} = {f.value}  (来源: {f.source}, 第 {f.step} 步)"
            for f in self._facts.values()
        ]
        return "已确认的事实（引用时请以此为准）:\n" + "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {"facts": len(self._facts), "conflicts": self.conflicts}
