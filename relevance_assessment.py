"""
模块③：论文相关性评估
使用 LLM 对候选论文与用户查询的相关性进行批量评分
"""
import json
from llm_config import chat_with_retry

# 相关性评估的系统提示词，定义三级评分标准
SYSTEM_PROMPT = """You are an academic paper relevance judge. Given a research question and a list of papers, classify each paper's relevance.

You must respond in EXACTLY this JSON format:
{
  "assessments": [
    {
      "paper_index": 0,
      "relevance": "highly_relevant" | "partially_relevant" | "not_relevant",
      "relevance_score": 0.85,
      "reason": "one sentence explanation"
    }
  ]
}

Classification criteria:
- "highly_relevant" (score 0.7-1.0): The paper directly addresses the core topic, methods, or problem in the user's query
- "partially_relevant" (score 0.3-0.7): The paper is in a related area or addresses only one aspect of the query
- "not_relevant" (score 0.0-0.3): The paper is tangentially related at best

Base your judgment primarily on the title and abstract. If the abstract is missing, be more conservative in your scoring."""

# 每批评估的论文数量，避免单次 LLM 调用过长
BATCH_SIZE = 10


def _format_papers(papers: list[dict], start_index: int = 0) -> str:
    """将论文列表格式化为文本，供 LLM 评估

    Args:
        papers: 论文列表
        start_index: 起始编号（用于批量处理时保持全局索引一致）

    Returns:
        格式化后的论文文本
    """
    lines = []
    for i, p in enumerate(papers):
        # 截断过长的摘要，控制 token 消耗
        abstract = (p.get("abstract") or "No abstract available")[:300]
        lines.append(
            f"[{start_index + i}] Title: {p['title']}\n"
            f"    Year: {p.get('year', 'N/A')} | Citations: {p.get('citationCount', 0)}\n"
            f"    Abstract: {abstract}"
        )
    return "\n\n".join(lines)


def assess_relevance(user_query: str, papers: list[dict]) -> list[dict]:
    """批量评估论文与用户查询的相关性

    Args:
        user_query: 用户的原始查询
        papers: 候选论文列表

    Returns:
        每篇论文新增三个字段：relevance（等级）、relevance_score（分数）、relevance_reason（理由）
    """
    if not papers:
        return papers

    # 分批调用 LLM，收集所有评估结果
    all_assessments = {}
    for batch_start in range(0, len(papers), BATCH_SIZE):
        batch = papers[batch_start : batch_start + BATCH_SIZE]
        papers_text = _format_papers(batch, start_index=batch_start)

        try:
            content = chat_with_retry(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Research question: {user_query}\n\nPapers:\n{papers_text}"},
                ],
                temperature=0.1,  # 低温度保证评分一致性
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            result = json.loads(content)
            for a in result.get("assessments", []):
                idx = a.get("paper_index", -1)
                if 0 <= idx < len(papers):
                    all_assessments[idx] = a
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [Assess] Batch assessment failed: {e}")

    # 将评估结果附加到原始论文数据上，未评估的默认为不相关
    for i, paper in enumerate(papers):
        a = all_assessments.get(i, {})
        paper["relevance"] = a.get("relevance", "not_relevant")
        paper["relevance_score"] = a.get("relevance_score", 0.0)
        paper["relevance_reason"] = a.get("reason", "")

    return papers


if __name__ == "__main__":
    test_papers = [
        {"title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "abstract": "We explore RAG...", "year": 2020, "citationCount": 5000},
        {"title": "Attention Is All You Need", "abstract": "We propose a new architecture...", "year": 2017, "citationCount": 100000},
    ]
    result = assess_relevance("RAG for LLMs", test_papers)
    for p in result:
        print(f"- [{p['relevance']}] {p['title']} (score: {p['relevance_score']})")
