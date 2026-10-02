"""RAG 测试：切块、检索排序、引用渲染、RAG-as-a-tool。"""

from __future__ import annotations

import pytest

from agentlab.rag import KnowledgeBase, build_search_tool, chunk_text

DOC = (
    "设备过热处置规程：当主轴温度在 10 分钟窗口内升幅超过 15 摄氏度，判定为温升异常。"
    "处置顺序为：降低主轴转速至额定值的 60%；检查冷却液流量是否低于 12 L/min；"
    "若 5 分钟内温度未回落，执行停机并通知设备工程师。"
)
OTHER = "振动异常处置规程：振动速度有效值超过 4.5 mm/s 时判定为异常，常见原因是轴承磨损与地脚螺栓松动。"


# --------------------------------------------------------------------------- #
# 切块
# --------------------------------------------------------------------------- #

def test_chunk_short_text_is_single_chunk():
    assert chunk_text("短文本", size=100, overlap=10) == ["短文本"]


def test_chunk_empty_text():
    assert chunk_text("   ", size=100, overlap=10) == []
    assert chunk_text("", size=100, overlap=10) == []


def test_chunk_covers_text_with_overlap():
    text = "。".join(f"第{i}段内容" + "x" * 20 for i in range(20))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks), "每块不得超过窗口大小"
    # 相邻块之间存在重叠：把两块拼起来应当找不到断裂的关键词
    assert chunks[0][-10:] in text
    joined = "".join(chunks)
    for marker in ("第0段", "第19段"):
        assert marker in joined


def test_chunk_prefers_paragraph_boundary():
    text = "A" * 60 + "\n" + "B" * 60
    chunks = chunk_text(text, size=80, overlap=10)
    assert chunks[0] == "A" * 60, "应当在换行处切开而不是硬切"


@pytest.mark.parametrize(
    "size, overlap",
    [(0, 0), (-1, 0), (10, 10), (10, 20), (10, -1)],
)
def test_chunk_rejects_bad_windows(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("内容" * 50, size=size, overlap=overlap)


# --------------------------------------------------------------------------- #
# 知识库
# --------------------------------------------------------------------------- #

def build_kb() -> KnowledgeBase:
    kb = KnowledgeBase(dim=256, size=200, overlap=40)
    kb.add_document("sop-overheat", DOC, metadata={"kind": "sop"})
    kb.add_document("sop-vibration", OTHER, metadata={"kind": "sop"})
    return kb


def test_add_document_creates_stable_chunk_ids():
    kb = KnowledgeBase(dim=128, size=40, overlap=10)
    chunks = kb.add_document("doc-a", "内容" * 100)
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks][:3] == ["doc-a#0", "doc-a#1", "doc-a#2"]
    assert all(c.doc_id == "doc-a" for c in chunks)
    assert kb.docs["doc-a"]["chunks"] == len(chunks)
    assert len(kb.docs["doc-a"]["sha256"]) == 16


def test_add_document_hashes_content():
    kb = KnowledgeBase(dim=64)
    kb.add_document("a", "同样的内容")
    kb.add_document("b", "同样的内容")
    assert kb.docs["a"]["sha256"] == kb.docs["b"]["sha256"]


def test_retrieve_returns_most_relevant_document_first():
    kb = build_kb()
    hits = kb.retrieve("温度升得太快怎么处理", top_k=2)
    assert hits and hits[0].doc_id == "sop-overheat"
    assert hits[0].index == 1
    assert hits[0].score > 0


def test_retrieve_respects_top_k_and_scores_descending():
    kb = build_kb()
    hits = kb.retrieve("规程", top_k=3)
    assert len(hits) <= 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert kb.queries == 1


def test_retrieve_on_empty_kb():
    assert KnowledgeBase().retrieve("任意问题") == []


def test_add_file(tmp_path):
    path = tmp_path / "sop.md"
    path.write_text(DOC, encoding="utf-8")
    kb = KnowledgeBase(dim=64)
    chunks = kb.add_file(path)
    assert chunks and chunks[0].doc_id == "sop"
    assert kb.retrieve("过热", top_k=1)


def test_render_context_has_numbered_citations():
    kb = build_kb()
    hits = kb.retrieve("温度异常处置", top_k=2)
    context = kb.render_context(hits)
    assert "【检索到的资料】" in context
    assert "[1]" in context
    assert "sop-overheat" in context
    assert "不要编造" in context


def test_render_context_empty_hits():
    assert KnowledgeBase().render_context([]) == ""


def test_render_context_truncates_to_budget():
    kb = KnowledgeBase(dim=64, size=1000, overlap=0)
    kb.add_document("big", "内容" * 500)
    hits = kb.retrieve("内容", top_k=1)
    context = kb.render_context(hits, max_chars=100)
    assert "…" in context
    assert len(context) < 400


def test_stats():
    kb = build_kb()
    kb.retrieve("温度")
    stats = kb.stats()
    assert stats["documents"] == 2
    assert stats["chunks"] >= 2
    assert stats["queries"] == 1
    assert stats["dim"] == 256


# --------------------------------------------------------------------------- #
# RAG-as-a-tool
# --------------------------------------------------------------------------- #

def test_search_tool_schema_and_behaviour():
    kb = build_kb()
    spec = build_search_tool(kb, top_k=2)
    assert spec.name == "search_knowledge_base"
    assert spec.parameters["required"] == ["query"]
    assert spec.parameters["properties"]["query"]["type"] == "string"

    result = spec.func(query="温度异常", k=1)
    assert len(result) == 1
    assert result[0]["cite"] == "[1]"
    assert result[0]["doc_id"] == "sop-overheat"
    assert 0.0 <= result[0]["score"] <= 1.0


def test_search_tool_clamps_k():
    kb = build_kb()
    spec = build_search_tool(kb)
    assert len(spec.func(query="规程", k=99)) <= 7
    assert len(spec.func(query="规程", k=0)) >= 1
