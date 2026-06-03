"""
模块④：Agent 调度策略（MVP 版本）
串联模块①②③，实现"多关键词并行搜索 → 去重合并 → 相关性评估 → 排序输出"的完整流程
"""
from api_wrapper import search_papers_parallel
from query_understanding import expand_query
from relevance_assessment import assess_relevance


def search(user_query: str, top_k: int = 10) -> dict:
    """执行完整的论文搜索流程

    Args:
        user_query: 用户的自然语言查询
        top_k: 最终返回的论文数量上限

    Returns:
        {"query_info": 查询信息, "papers": 排序后的论文列表}
    """
    # 第一步：调用模块②，用 LLM 理解查询并生成多组搜索词
    print(f"[Agent] Expanding query: \"{user_query}\"")
    query_info = expand_query(user_query)
    search_queries = query_info["search_queries"]
    print(f"[Agent] Generated {len(search_queries)} search queries: {search_queries}")

    # 第二步：并行搜索所有查询词，自动去重合并
    print(f"[Agent] Parallel searching {len(search_queries)} queries...")
    all_papers = search_papers_parallel(search_queries, limit=15)
    print(f"[Agent] Total unique papers: {len(all_papers)}")

    if not all_papers:
        return {"query_info": query_info, "papers": []}

    # 第三步：调用模块③，用 LLM 批量评估每篇论文的相关性
    print(f"[Agent] Assessing relevance of {len(all_papers)} papers...")
    all_papers = assess_relevance(user_query, all_papers)

    # 第四步：过滤不相关论文，按相关性分数和引用数排序，取 top_k
    relevant_papers = [p for p in all_papers if p["relevance"] != "not_relevant"]
    relevant_papers.sort(key=lambda p: (-p.get("relevance_score", 0), -(p.get("citationCount") or 0)))
    relevant_papers = relevant_papers[:top_k]

    highly = sum(1 for p in relevant_papers if p["relevance"] == "highly_relevant")
    partial = sum(1 for p in relevant_papers if p["relevance"] == "partially_relevant")
    print(f"[Agent] Results: {highly} highly_relevant, {partial} partially_relevant (top {len(relevant_papers)})")

    return {"query_info": query_info, "papers": relevant_papers}
