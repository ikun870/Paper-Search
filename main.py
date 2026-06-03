"""
主入口：加载配置 → 接收查询 → 执行搜索 → 展示和保存结果
"""
import os
import sys
import time
from dotenv import load_dotenv
from agent_strategy import search
from output_formatter import format_results, save_results


def main():
    # 加载 .env 中的环境变量（API Key 等）
    load_dotenv()

    # 校验 LLM 配置是否完整
    if not os.getenv("LLM_API_KEY") or not os.getenv("LLM_BASE_URL") or not os.getenv("LLM_MODEL"):
        print("Error: 请在 .env 中配置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL")
        sys.exit(1)

    # 支持命令行参数传入查询，或交互式输入
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your research question: ").strip()

    if not query:
        print("Error: empty query")
        sys.exit(1)

    print(f"\nSearching for: {query}\n")

    # 执行搜索流程并计时
    start = time.time()
    raw = search(query, top_k=10)
    output = format_results(raw["query_info"], raw["papers"])
    elapsed = time.time() - start

    # 打印搜索结果摘要
    print(f"\n{'='*60}")
    print(f"Results for: {query}")
    print(f"{'='*60}")
    print(f"Total found: {output['summary']['total_found']}")
    print(f"Highly relevant: {output['summary']['highly_relevant']}")
    print(f"Partially relevant: {output['summary']['partially_relevant']}")
    print(f"Time: {elapsed:.1f}s")

    # 按相关性等级打印前 5 篇论文的详细信息
    for category in ["highly_relevant", "partially_relevant"]:
        papers = output["results"].get(category, [])
        if papers:
            label = category.replace("_", " ").title()
            print(f"\n--- {label} ---")
            for i, p in enumerate(papers[:5], 1):
                print(f"\n{i}. {p['title']}")
                authors = p["authors"][:3]
                suffix = "..." if len(p["authors"]) > 3 else ""
                print(f"   Authors: {', '.join(authors)}{suffix}")
                print(f"   Year: {p['year']} | Citations: {p['citationCount']}")
                print(f"   Score: {p['relevance_score']:.2f} | {p['relevance_reason']}")
                if p.get("abstract"):
                    print(f"   Abstract: {p['abstract'][:150]}...")
                print(f"   URL: {p['url']}")

    # 将完整结果保存为 JSON 文件
    output_file = f"output/results_{int(time.time())}.json"
    save_results(output, output_file)
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
