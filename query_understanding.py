"""
模块②：查询理解与扩展
使用 LLM 解析用户查询意图，提取关键实体，生成多组搜索词以提升召回率
"""
import json
from llm_config import chat_with_retry

# 查询扩展的系统提示词，要求 LLM 输出结构化 JSON
SYSTEM_PROMPT = """You are an academic research assistant. Analyze the user's natural language research question and generate multiple effective search queries for academic paper databases.

You must respond in EXACTLY this JSON format, with no other text:
{
  "entities": {
    "methods": ["list of research methods or techniques mentioned"],
    "domains": ["list of academic domains or fields mentioned"],
    "key_terms": ["list of other important technical terms"]
  },
  "search_queries": [
    "query 1: direct keyword search",
    "query 2: rephrased with synonyms",
    "query 3: broader domain-level search",
    "query 4: focused on specific methods"
  ]
}

Rules:
- Generate exactly 3 to 5 search_queries
- Each query should be 2-8 words, optimized for keyword search (no sentences, no question marks)
- Queries should cover different angles: specific terms, broader context, alternative terminology
- Prioritize terms that would appear in paper titles and abstracts"""


def expand_query(user_query: str) -> dict:
    """使用 LLM 解析查询意图并生成多组搜索词

    Args:
        user_query: 用户的自然语言查询

    Returns:
        {
            "original_query": 原始查询,
            "entities": {methods: [], domains: [], key_terms: []},
            "search_queries": [搜索词1, 搜索词2, ...]
        }
    """
    try:
        content = chat_with_retry(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        result = json.loads(content)
        # 确保原始查询始终包含在搜索词列表中作为兜底
        if user_query not in result.get("search_queries", []):
            result["search_queries"].append(user_query)
        return {
            "original_query": user_query,
            "entities": result.get("entities", {}),
            "search_queries": result.get("search_queries", [user_query]),
        }
    except (json.JSONDecodeError, Exception) as e:
        # LLM 输出解析失败时，用原始查询兜底
        print(f"  [Query] Expansion failed: {e}, using original query")
        return {
            "original_query": user_query,
            "entities": {},
            "search_queries": [user_query],
        }


if __name__ == "__main__":
    result = expand_query("What are the recent advances in retrieval-augmented generation for large language models?")
    print(json.dumps(result, indent=2, ensure_ascii=False))
