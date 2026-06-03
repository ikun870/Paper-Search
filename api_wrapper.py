"""
模块①：学术搜索 API 封装
支持 Semantic Scholar 和 OpenAlex 双源并行竞速：
  - 有 S2 API Key 时，两个 API 同时发请求，谁先回用谁
  - 无 S2 API Key 时，直接用 OpenAlex，跳过 S2 避免限流浪费时间
  - 支持多组搜索词并行请求
"""
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Semantic Scholar 论文搜索接口
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# OpenAlex 论文搜索接口
OPENALEX_URL = "https://api.openalex.org/works"
# 需要从 API 获取的论文字段
FIELDS = "title,abstract,year,authors,citationCount,url"
MAX_RETRIES = 3


def _has_s2_key() -> bool:
    """检查是否配置了 Semantic Scholar API Key"""
    return bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY"))


def search_papers(query: str, limit: int = 20) -> list[dict]:
    """搜索单个查询，双源竞速取最快结果

    Args:
        query: 搜索关键词
        limit: 返回论文数量上限

    Returns:
        统一格式的论文列表
    """
    # 无 S2 Key 时直接走 OpenAlex，跳过必定限流的 S2
    if not _has_s2_key():
        return _search_openalex(query, limit)

    # 有 S2 Key 时，两个 API 并行竞速，谁先返回非空结果用谁
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_search_semantic_scholar, query, limit): "S2",
            pool.submit(_search_openalex, query, limit): "OpenAlex",
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                results = future.result()
                if results:
                    return results
            except Exception:
                continue
    return []


def search_papers_parallel(queries: list[str], limit: int = 15) -> list[dict]:
    """并行搜索多组查询词，合并去重

    Args:
        queries: 多组搜索关键词
        limit: 每组查询返回的论文数量上限

    Returns:
        去重后的论文列表
    """
    all_papers = []
    seen_ids = set()

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {pool.submit(search_papers, q, limit): q for q in queries}
        for future in as_completed(futures):
            query = futures[future]
            try:
                papers = future.result()
                new_count = 0
                for p in papers:
                    pid = p["paperId"]
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        all_papers.append(p)
                        new_count += 1
                print(f"  [API] \"{query[:40]}\" → {len(papers)} found, {new_count} new")
            except Exception as e:
                print(f"  [API] \"{query[:40]}\" failed: {e}")

    return all_papers


def _search_semantic_scholar(query: str, limit: int) -> list[dict]:
    """调用 Semantic Scholar API 搜索论文，支持限流重试"""
    papers = []
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                SEMANTIC_SCHOLAR_URL,
                params={"query": query, "limit": limit, "fields": FIELDS},
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  [API] S2 rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            for raw in resp.json().get("data", []):
                authors = [
                    a["name"] for a in (raw.get("authors") or [])
                    if isinstance(a, dict) and a.get("name")
                ]
                papers.append({
                    "paperId": raw.get("paperId", ""),
                    "title": raw.get("title", ""),
                    "abstract": raw.get("abstract"),
                    "year": raw.get("year"),
                    "authors": authors,
                    "citationCount": raw.get("citationCount", 0),
                    "url": raw.get("url", ""),
                })
            return papers
        except requests.RequestException as e:
            print(f"  [API] S2 request failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
    return papers


def _search_openalex(query: str, limit: int) -> list[dict]:
    """调用 OpenAlex API 搜索论文"""
    papers = []
    try:
        resp = requests.get(
            OPENALEX_URL,
            params={
                "search": query,
                "per_page": limit,
                "select": "id,title,abstract_inverted_index,publication_year,authorships,cited_by_count",
            },
            headers={"User-Agent": "PaperSearchBot/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        for raw in resp.json().get("results", []):
            abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))
            authors = [
                a["author"]["display_name"]
                for a in (raw.get("authorships") or [])
                if a.get("author", {}).get("display_name")
            ]
            papers.append({
                "paperId": raw.get("id", ""),
                "title": raw.get("title", ""),
                "abstract": abstract,
                "year": raw.get("publication_year"),
                "authors": authors,
                "citationCount": raw.get("cited_by_count", 0),
                "url": f"https://openalex.org/works/{raw.get('id', '')}",
            })
    except requests.RequestException as e:
        print(f"  [API] OpenAlex request failed: {e}")
    return papers


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """将 OpenAlex 的倒排索引摘要重建为完整文本

    OpenAlex 返回的摘要是 {"word": [position1, position2, ...]} 格式，
    需要按位置排序后拼接还原
    """
    if not inverted_index:
        return None
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


if __name__ == "__main__":
    results = search_papers("retrieval augmented generation", limit=5)
    for p in results:
        print(f"- [{p['year']}] {p['title']} (citations: {p['citationCount']})")
