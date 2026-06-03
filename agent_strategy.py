"""
模块④：Agent 调度策略（v2 迭代版本）
实现多轮迭代搜索，模拟人类研究员"搜索 → 阅读 → 顺藤摸瓜 → 再搜索"的完整流程

核心改进（vs MVP 单轮版）：
  - 每轮评估相关性后，对高分论文展开参考文献（Expand 动作）
  - 新发现的论文加入队列，下一轮继续评估
  - 自动判断停止条件：收敛、论文数上限、最大轮次
"""
from api_wrapper import search_papers_parallel, expand_papers
from query_understanding import expand_query
from relevance_assessment import assess_relevance


def search(
    user_query: str,
    top_k: int = 10,
    max_rounds: int = 3,
    max_papers: int = 100,
    expand_top_n: int = 5,
) -> dict:
    """执行迭代式论文搜索流程

    每轮迭代做的事情：
        1. 评估当前队列中尚未评估的论文
        2. 找出高分论文（highly_relevant），展开它们的参考文献
        3. 新论文加入队列，下一轮继续
        4. 检查停止条件

    Args:
        user_query: 用户的自然语言查询
        top_k: 最终返回的论文数量上限
        max_rounds: 最大迭代轮次（防止无限循环）
        max_papers: 累计论文数量上限（控制 API 成本）
        expand_top_n: 每轮展开前 N 篇高分论文的引用

    Returns:
        {"query_info": 查询信息, "papers": 排序后的论文列表}
    """
    # ═══ 第一步：LLM 查询理解，生成多组搜索词 ═══
    print(f"[Agent] Expanding query: \"{user_query}\"")
    query_info = expand_query(user_query)
    search_queries = query_info["search_queries"]
    print(f"[Agent] Generated {len(search_queries)} search queries: {search_queries}")

    # ═══ 第二步：初始搜索 ═══
    print(f"[Agent] Round 1: Initial search with {len(search_queries)} queries...")
    all_papers = search_papers_parallel(search_queries, limit=15)
    seen_ids = {p["paperId"] for p in all_papers if p["paperId"]}
    print(f"[Agent] Initial search: {len(all_papers)} unique papers")

    if not all_papers:
        return {"query_info": query_info, "papers": []}

    # ═══ 第三步：迭代循环 —— 评估 → 展开 → 再评估 ═══
    for round_num in range(1, max_rounds + 1):
        # 3a. 评估尚未评分的论文
        unassessed = [p for p in all_papers if "relevance" not in p]
        if unassessed:
            print(f"[Agent] Round {round_num}: Assessing {len(unassessed)} new papers...")
            assess_relevance(user_query, unassessed)

        # 3b. 找出高分论文，准备展开引用
        highly = [
            p for p in all_papers
            if p.get("relevance") == "highly_relevant" and p.get("paperId")
        ]
        highly.sort(key=lambda p: -p.get("relevance_score", 0))

        # 只展开当前轮次新标记为 highly_relevant 的论文（避免重复展开）
        newly_highly = [
            p for p in highly[:expand_top_n]
            if not p.get("_expanded")
        ]

        if not newly_highly:
            print(f"[Agent] Round {round_num}: No new highly_relevant papers to expand, search converged.")
            break

        # 3c. 展开高分论文的参考文献（这是 PaSa 的 Expand 动作）
        print(f"[Agent] Round {round_num}: Expanding references from top {len(newly_highly)} papers...")
        for p in newly_highly:
            print(f"  [Agent]   Expand: {p['title'][:80]}... (score={p.get('relevance_score', 0):.2f})")
            p["_expanded"] = True  # 标记已展开，避免重复

        expand_ids = [p["paperId"] for p in newly_highly]
        new_papers = expand_papers(expand_ids, limit=20)

        # 3d. 过滤已见过的论文，加入队列
        truly_new = [p for p in new_papers if p["paperId"] not in seen_ids]
        for p in truly_new:
            seen_ids.add(p["paperId"])
            all_papers.append(p)

        print(f"[Agent] Round {round_num}: {len(new_papers)} refs found, "
              f"{len(truly_new)} new → total {len(all_papers)} papers in queue")

        # 3e. 检查停止条件
        if not truly_new:
            print(f"[Agent] No new papers discovered, search converged.")
            break

        if len(all_papers) >= max_papers:
            print(f"[Agent] Reached max_papers limit ({max_papers}), stopping search.")
            break

    # ═══ 第四步：排序输出 ═══
    # 过滤 not_relevant，按 relevance_score 和引用数降序排列
    relevant = [p for p in all_papers if p.get("relevance") != "not_relevant"]
    relevant.sort(key=lambda p: (-p.get("relevance_score", 0), -(p.get("citationCount") or 0)))
    final_papers = relevant[:top_k]

    highly = sum(1 for p in final_papers if p.get("relevance") == "highly_relevant")
    partial = sum(1 for p in final_papers if p.get("relevance") == "partially_relevant")
    print(f"[Agent] Final: {highly} highly_relevant, {partial} partially_relevant "
          f"(top {len(final_papers)} from {len(all_papers)} total)")

    # 清理内部标记字段
    for p in final_papers:
        p.pop("_expanded", None)

    return {"query_info": query_info, "papers": final_papers}
