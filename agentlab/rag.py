"""RAG：文档切块、向量检索、引用溯源。

对应 JD 职责里的「落地 RAG …… 提升可靠性、可控性」。
这里的取舍是**引用可溯源优先于召回率**：

- 每个 chunk 都有稳定 ID（``doc_id#序号``），检索结果带 ``score`` 与来源；
- :meth:`KnowledgeBase.render_context` 输出带 ``[1] [2]`` 编号的上下文，
  并附一段「只允许引用编号内的事实」的约束说明；
- 检索可以被注册成**工具**（:func:`build_search_tool`），
  这样 Agent 既可以在规划阶段主动检索，也可以在工具循环里按需检索
  —— 也就是「RAG 作为工具」而不是「一次性拼接上下文」。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .memory import VectorMemory


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cite(self) -> str:
        return f"[{self.index}] {self.doc_id} ({self.score:.3f})"


def chunk_text(text: str, size: int = 240, overlap: int = 48) -> List[str]:
    """按字符窗口切块，带重叠，并且**尽量不切断段落**。

    优先在窗口内找最后一个换行/句末标点作为切点，找不到才硬切。
    重叠区保证跨块的句子不会被检索漏掉。
    """
    if size <= 0:
        raise ValueError("size 必须为正数")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")
    clean = (text or "").strip()
    if not clean:
        return []
    if len(clean) <= size:
        return [clean]

    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        if end < len(clean):
            window = clean[start:end]
            cut = max(window.rfind("\n"), window.rfind("。"), window.rfind("；"), window.rfind(". "))
            if cut > size * 0.5:
                end = start + cut + 1
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


class KnowledgeBase:
    """基于 :class:`VectorMemory` 的极简知识库（零第三方依赖）。"""

    def __init__(self, dim: int = 512, size: int = 240, overlap: int = 48) -> None:
        self.memory = VectorMemory(dim=dim)
        self.size = size
        self.overlap = overlap
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.queries = 0

    def __len__(self) -> int:
        return len(self.memory)

    # -- 入库 ------------------------------------------------------------- #
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        pieces = chunk_text(text, self.size, self.overlap)
        chunks: List[Chunk] = []
        for idx, piece in enumerate(pieces):
            chunk = Chunk(
                chunk_id=f"{doc_id}#{idx}",
                doc_id=doc_id,
                text=piece,
                index=idx,
                metadata=dict(metadata or {}),
            )
            self.memory.add(
                piece,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "chunk_id": chunk.chunk_id,
                    **(metadata or {}),
                },
                item_id=chunk.chunk_id,
            )
            chunks.append(chunk)
        self.docs[doc_id] = {
            "chunks": len(chunks),
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "metadata": dict(metadata or {}),
        }
        return chunks

    def add_file(self, path: Any, doc_id: Optional[str] = None, encoding: str = "utf-8") -> List[Chunk]:
        p = Path(path)
        text = p.read_text(encoding=encoding)
        return self.add_document(doc_id or p.stem, text, metadata={"path": str(p.name)})

    # -- 检索 ------------------------------------------------------------- #
    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Hit]:
        self.queries += 1
        raw = self.memory.search(query, top_k=top_k, min_score=min_score)
        hits: List[Hit] = []
        for rank, item in enumerate(raw, start=1):
            hits.append(
                Hit(
                    chunk_id=item.item_id,
                    doc_id=str(item.metadata.get("doc_id") or "unknown"),
                    text=item.text,
                    score=item.score,
                    index=rank,
                    metadata=dict(item.metadata),
                )
            )
        return hits

    def render_context(self, hits: Sequence[Hit], max_chars: int = 1600) -> str:
        """把检索结果渲染成带编号的上下文，供提示词使用。"""
        if not hits:
            return ""
        lines = ["【检索到的资料】"]
        used = 0
        for hit in hits:
            body = hit.text.replace("\n", " ").strip()
            if used + len(body) > max_chars:
                body = body[: max(0, max_chars - used)] + "…"
            lines.append(f"[{hit.index}] 来源 {hit.doc_id}（相似度 {hit.score:.3f}）: {body}")
            used += len(body)
            if used >= max_chars:
                break
        lines.append("回答时若引用上述内容，请标注对应的编号，不要编造编号以外的信息。")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {
            "documents": len(self.docs),
            "chunks": len(self.memory),
            "queries": self.queries,
            "dim": self.memory.dim,
        }


def build_search_tool(kb: KnowledgeBase, top_k: int = 3, tags: Sequence[str] = ("检索", "rag")):
    """把知识库检索注册成一个工具（RAG-as-a-tool）。"""

    def search_knowledge_base(query: str, k: int = top_k) -> List[Dict[str, Any]]:
        """在本地知识库中检索与问题相关的资料片段，返回带来源与相似度的条目。

        Args:
            query: 自然语言检索问题。
            k: 返回的片段数量，1~7。
        """
        k = max(1, min(int(k), 7))
        hits = kb.retrieve(query, top_k=k)
        return [
            {
                "cite": f"[{h.index}]",
                "doc_id": h.doc_id,
                "score": round(h.score, 4),
                "text": h.text[:400],
            }
            for h in hits
        ]

    from .tools import _build_spec  # 局部导入避免循环依赖

    return _build_spec(search_knowledge_base, name="search_knowledge_base", tags=tags)
