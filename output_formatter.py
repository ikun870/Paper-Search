"""
模块⑤：结构化输出格式化
将搜索结果整理为分类清晰的 JSON 格式，并支持保存到文件
"""
import json
import os


def format_results(query_info: dict, papers: list[dict]) -> dict:
    """将搜索结果格式化为结构化输出

    Args:
        query_info: 查询信息（原始查询、实体、扩展搜索词）
        papers: 经过相关性评估和排序的论文列表

    Returns:
        结构化输出字典，包含查询信息、统计摘要和分类结果
    """
    # 按相关性等级分组
    highly = [p for p in papers if p.get("relevance") == "highly_relevant"]
    partial = [p for p in papers if p.get("relevance") == "partially_relevant"]

    # 清洗论文数据，只保留输出需要的字段
    def _clean(paper):
        return {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract"),
            "year": paper.get("year"),
            "authors": paper.get("authors", []),
            "citationCount": paper.get("citationCount", 0),
            "url": paper.get("url", ""),
            "relevance_score": paper.get("relevance_score", 0),
            "relevance_reason": paper.get("relevance_reason", ""),
        }

    return {
        "query": {
            "original": query_info.get("original_query", ""),
            "entities": query_info.get("entities", {}),
            "expanded_queries": query_info.get("search_queries", []),
        },
        "summary": {
            "total_found": len(papers),
            "highly_relevant": len(highly),
            "partially_relevant": len(partial),
        },
        "results": {
            "highly_relevant": [_clean(p) for p in highly],
            "partially_relevant": [_clean(p) for p in partial],
        },
    }


def save_results(output: dict, filepath: str):
    """将结果保存为 JSON 文件

    Args:
        output: format_results 返回的结构化输出
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
