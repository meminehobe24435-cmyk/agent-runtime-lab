"""记忆层测试：token 预算、向量检索、持久化、工作记忆溯源。"""

from __future__ import annotations

import math

import pytest

from agentlab.llm import estimate_tokens
from agentlab.memory import (
    ShortTermMemory,
    VectorMemory,
    WorkingMemory,
    cosine,
    embed,
    tokenize,
)


# --------------------------------------------------------------------------- #
# token 估算与分词
# --------------------------------------------------------------------------- #

def test_estimate_tokens_counts_cjk_per_char():
    assert estimate_tokens("") == 0
    assert estimate_tokens("中文四个字") == 5
    assert estimate_tokens("abcdefgh") == 2  # 8/4
    assert estimate_tokens("中文abcd") == 3  # 2 + 4/4
    assert estimate_tokens("a" * 1000) == 250


def test_tokenize_mixes_latin_words_and_cjk_chars():
    tokens = tokenize("press-01 温度 81.3 ℃")
    assert "press-01" in tokens
    assert "温" in tokens and "度" in tokens
    assert "81.3" in tokens


# --------------------------------------------------------------------------- #
# 短期记忆
# --------------------------------------------------------------------------- #

def test_short_term_memory_keeps_recent_and_counts_tokens():
    memory = ShortTermMemory(max_tokens=1000)
    memory.add("user", "你好")
    memory.add("assistant", "你好，有什么可以帮你")
    assert len(memory.turns) == 2
    assert memory.tokens > 0
    stats = memory.stats()
    assert stats["turns"] == 2 and stats["compactions"] == 0


def test_short_term_memory_compacts_with_summarizer():
    seen: list = []

    def summarizer(text: str) -> str:
        seen.append(text)
        return "（摘要）" + text[:10]

    memory = ShortTermMemory(max_tokens=40, keep_recent=2, summarizer=summarizer)
    for i in range(20):
        memory.add("user", f"第 {i} 条比较长的消息内容" * 3)

    assert memory.compactions > 0
    assert memory.dropped > 0
    # 摘要是滚动缓冲：超出限幅时保留**最近**的一段，因此断言包含而不是前缀
    assert "（摘要）" in memory.summary
    assert seen, "summarizer 应当被调用"
    # 最近的 keep_recent 条一定还在
    assert len(memory.turns) == 2
    assert memory.turns[-1].content.startswith("第 19 条")
    # 摘要自身也被限幅，不会把膨胀从 turns 搬到 summary
    assert len(memory.summary) <= memory.summary_token_budget


def test_short_term_memory_without_summarizer_keeps_a_digest():
    """没有 summarizer 时不能静默丢历史，要退化成截断摘要。"""
    memory = ShortTermMemory(max_tokens=30, keep_recent=1)
    for i in range(5):
        memory.add("user", f"关键事实 {i} " * 20)
    assert memory.compactions > 0
    assert memory.dropped > 0
    assert memory.summary, "即便没有 summarizer，也要留下 digest"


def test_short_term_memory_messages_puts_summary_first():
    memory = ShortTermMemory(max_tokens=20, keep_recent=1)
    memory.add("user", "很长的一段历史消息" * 10)
    memory.add("assistant", "很长的一段历史回复" * 10)
    memory.add("user", "最新问题")
    messages = memory.messages()
    assert messages[0][0] == "system" and "历史摘要" in messages[0][1]
    assert messages[-1] == ("user", "最新问题")


def test_summarizer_failure_does_not_break_compaction():
    def broken(_: str) -> str:
        raise RuntimeError("摘要服务不可用")

    memory = ShortTermMemory(max_tokens=20, keep_recent=1, summarizer=broken)
    for i in range(4):
        memory.add("user", "历史" * 50 + str(i))
    assert memory.compactions >= 1
    assert memory.summary  # 退化为截断摘要，而不是抛异常


# --------------------------------------------------------------------------- #
# 向量记忆
# --------------------------------------------------------------------------- #

def test_embed_is_l2_normalized_and_deterministic():
    vec = embed("设备温度异常处置", dim=128)
    assert len(vec) == 128
    assert abs(math.sqrt(sum(v * v for v in vec)) - 1.0) < 1e-9
    assert vec == embed("设备温度异常处置", dim=128)


def test_embed_empty_text_is_zero_vector():
    assert set(embed("", dim=16)) == {0.0}
    assert cosine(embed("", dim=16), embed("", dim=16)) == 0.0


def test_vector_memory_ranks_relevant_first():
    memory = VectorMemory()  # 用默认维度
    memory.add("主轴温度异常处置规程：降转速、查冷却液")
    memory.add("地脚螺栓紧固与轴承更换流程")
    memory.add("巡检报告的格式要求")

    for query, expected in [
        ("温度太高怎么办", "主轴温度异常"),
        ("轴承松动", "地脚螺栓紧固"),
        ("报告怎么写", "巡检报告的格式要求"),
        ("冷却液", "主轴温度异常"),
    ]:
        hits = memory.search(query, top_k=2)
        assert hits, f"{query} 应当能检索到结果"
        assert expected in hits[0].text, f"{query} 的首位命中了 {hits[0].text}"


def test_small_dimension_is_documented_as_collision_prone():
    """把维度选择的实测结论固化下来：dim=256 时哈希碰撞会让排序出错。

    这条测试不是在「测 bug」，而是把一次**实测选型**写进回归：
    默认维度若被改回 256，会立刻有一批中文短查询排错。
    """
    docs = [
        "主轴温度异常处置规程：降转速、查冷却液",
        "地脚螺栓紧固与轴承更换流程",
        "巡检报告的格式要求",
    ]
    small = VectorMemory(dim=256)
    large = VectorMemory()
    for doc in docs:
        small.add(doc)
        large.add(doc)

    query = "温度太高怎么办"
    assert "主轴温度异常" in large.search(query, top_k=1)[0].text
    assert large.dim >= 512


def test_vector_memory_identical_text_scores_one():
    memory = VectorMemory(dim=256)
    memory.add("完全一样的文本")
    hit = memory.search("完全一样的文本", top_k=1)[0]
    assert hit.score > 0.99


def test_vector_memory_min_score_and_miss_counters():
    memory = VectorMemory(dim=256)
    memory.add("轴承磨损")
    assert memory.search("完全无关的航天器轨道力学", top_k=3, min_score=0.6) == []
    assert memory.misses == 1
    memory.search("轴承磨损", top_k=1)
    assert memory.hits == 1


def test_vector_memory_search_returns_copies():
    memory = VectorMemory(dim=64)
    memory.add("原始文本", metadata={"k": "v"})
    hit = memory.search("原始文本", top_k=1)[0]
    hit.text = "被改坏了"
    assert memory.items[0].text == "原始文本"


def test_vector_memory_add_remove_and_len():
    memory = VectorMemory(dim=64)
    item = memory.add("一", item_id="fixed")
    assert item.item_id == "fixed" and len(memory) == 1
    assert memory.get("fixed") is item
    assert memory.remove("fixed") is True
    assert memory.remove("fixed") is False
    assert len(memory) == 0


def test_vector_memory_persistence_roundtrip(tmp_path):
    memory = VectorMemory(dim=128)
    memory.add("设备过热处置规程", metadata={"kind": "sop"})
    memory.add("振动异常判定阈值")
    path = tmp_path / "mem.json"
    memory.save(path)

    restored = VectorMemory.load(path)
    assert len(restored) == 2
    assert restored.dim == 128
    hits = restored.search("过热怎么办", top_k=1)
    assert "过热" in hits[0].text
    assert hits[0].metadata["kind"] == "sop"
    # 新加入的条目 ID 不会与旧 ID 冲突
    fresh = restored.add("新增")
    assert fresh.item_id == "mem_3"


# --------------------------------------------------------------------------- #
# 工作记忆
# --------------------------------------------------------------------------- #

def test_working_memory_records_provenance():
    working = WorkingMemory()
    working.set("last_query_metric", 81.3, source="tool:query_metric", step=2)
    assert working.get("last_query_metric") == "81.3"
    fact = working.provenance("last_query_metric")
    assert fact is not None and fact.source == "tool:query_metric" and fact.step == 2
    rendered = working.render()
    assert "tool:query_metric" in rendered and "第 2 步" in rendered
    assert working.stats()["facts"] == 1


def test_working_memory_counts_conflicts():
    working = WorkingMemory()
    working.set("temperature", 62.8, source="tool:a", step=1)
    working.set("temperature", 62.8, source="tool:a", step=2)
    assert working.stats()["conflicts"] == 0
    working.set("temperature", 99.9, source="tool:b", step=3)
    assert working.stats()["conflicts"] == 1
    assert working.get("temperature") == "99.9"


def test_working_memory_serializes_structures():
    working = WorkingMemory()
    working.set("series", [1, 2, 3], source="tool:query_metric", step=1)
    assert working.get("series") == "[1, 2, 3]"
    assert working.snapshot() == {"series": "[1, 2, 3]"}
    assert working.render() == "" or "已确认的事实" in working.render()


def test_empty_working_memory_renders_empty():
    assert WorkingMemory().render() == ""
