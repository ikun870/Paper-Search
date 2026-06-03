"""
评测脚本：加载数据集 → 运行 Agent → 计算 Recall@k / Precision@k / F1
支持 AutoScholarQuery 和 RealScholarQuery 格式（JSONL，每行一个 query）

用法：
    python evaluate.py <dataset.jsonl>              # 评测全部查询
    python evaluate.py <dataset.jsonl> 10           # 只评测前 10 条
    python evaluate.py --create-sample              # 生成样例数据集
"""
import json
import os
import sys
import time
from dotenv import load_dotenv
from agent_strategy import search


def load_dataset(filepath: str) -> list[dict]:
    """加载 JSONL 格式的数据集

    每行格式：
        {"query": "研究问题", "relevant_papers": ["paper_id_1", ...]}

    Args:
        filepath: JSONL 文件路径

    Returns:
        查询列表
    """
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def compute_metrics(
    agent_paper_ids: list[str],
    ground_truth_ids: list[str],
    k_values: list[int] = [5, 10, 20, 50, 100],
) -> dict:
    """计算 Recall@k 和 Precision@k

    Recall@k = |top_k ∩ ground_truth| / |ground_truth|
    Precision@k = |top_k ∩ ground_truth| / k

    Args:
        agent_paper_ids: Agent 返回的论文 ID 列表（已排序）
        ground_truth_ids: 标注的相关论文 ID 列表
        k_values: 需要计算的 k 值

    Returns:
        包含各指标值的字典
    """
    metrics = {}
    gt_set = set(ground_truth_ids)
    gt_count = len(gt_set)

    if gt_count == 0:
        # 无标注数据时，返回空指标
        for k in k_values:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"precision@{k}"] = 0.0
        metrics["f1@20"] = 0.0
        metrics["f1@50"] = 0.0
        return metrics

    for k in k_values:
        top_k = set(agent_paper_ids[:k])
        hits = len(top_k & gt_set)
        recall = hits / gt_count
        precision = hits / min(k, len(top_k)) if top_k else 0.0
        metrics[f"recall@{k}"] = round(recall, 4)
        metrics[f"precision@{k}"] = round(precision, 4)

    # 计算 F1@20 和 F1@50（比赛的两个核心指标）
    for k in [20, 50]:
        r = metrics.get(f"recall@{k}", 0)
        p = metrics.get(f"precision@{k}", 0)
        if r + p > 0:
            metrics[f"f1@{k}"] = round(2 * p * r / (p + r), 4)
        else:
            metrics[f"f1@{k}"] = 0.0

    return metrics


def run_evaluation(dataset_path: str, max_queries: int = None) -> dict:
    """运行完整评测

    Args:
        dataset_path: 数据集 JSONL 文件路径
        max_queries: 最多评测几条查询（None = 全部）

    Returns:
        包含每条查询结果和汇总指标的字典
    """
    print(f"[LOAD] Loading dataset: {dataset_path}")
    queries = load_dataset(dataset_path)

    if max_queries:
        queries = queries[:max_queries]

    print(f"[EVAL] Evaluating {len(queries)} queries...\n")

    all_metrics = []
    per_query_results = []

    for i, item in enumerate(queries):
        user_query = item.get("query", "")
        ground_truth = item.get("relevant_papers", [])

        print(f"\n{'═' * 60}")
        print(f"[{i+1}/{len(queries)}] {user_query[:120]}")
        print(f"  Ground truth papers: {len(ground_truth)}")

        start = time.time()
        try:
            result = search(user_query, top_k=100)
            papers = result.get("papers", [])
            agent_ids = [p.get("paperId", "") for p in papers if p.get("paperId")]
        except Exception as e:
            print(f"  [ERROR] {e}")
            papers = []
            agent_ids = []

        elapsed = time.time() - start

        metrics = compute_metrics(agent_ids, ground_truth)
        metrics["query"] = user_query[:100]
        metrics["time_s"] = round(elapsed, 1)
        metrics["papers_found"] = len(papers)
        metrics["gt_count"] = len(ground_truth)

        all_metrics.append(metrics)
        per_query_results.append({
            "query": user_query,
            "ground_truth": ground_truth,
            "agent_papers": agent_ids,
            "metrics": metrics,
        })

        print(f"  Recall@20: {metrics['recall@20']:.2%}  "
              f"Recall@50: {metrics['recall@50']:.2%}  "
              f"Precision@20: {metrics['precision@20']:.2%}  "
              f"F1@20: {metrics['f1@20']:.2%}  "
              f"Time: {elapsed:.1f}s")

    # ═══ 汇总统计 ═══
    avg_metrics = {}
    numeric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
    for key in numeric_keys:
        avg_metrics[key] = round(sum(m[key] for m in all_metrics) / len(all_metrics), 4)

    print(f"\n{'═' * 60}")
    print("=== EVALUATION SUMMARY ===")
    print(f"{'═' * 60}")
    print(f"  Queries evaluated:   {len(queries)}")
    print(f"  Avg Recall@20:       {avg_metrics['recall@20']:.2%}")
    print(f"  Avg Recall@50:       {avg_metrics['recall@50']:.2%}")
    print(f"  Avg Precision@20:    {avg_metrics['precision@20']:.2%}")
    print(f"  Avg F1@20:           {avg_metrics['f1@20']:.2%}")
    print(f"  Avg F1@50:           {avg_metrics['f1@50']:.2%}")
    print(f"  Avg Time per query:  {avg_metrics['time_s']:.1f}s")
    print(f"  Avg Papers found:    {avg_metrics['papers_found']:.1f}")
    print(f"  Avg GT size:         {avg_metrics['gt_count']:.1f}")

    # 比赛要求检查
    print(f"\n{'─' * 40}")
    f1_20 = avg_metrics['f1@20']
    if f1_20 >= 0.70:
        print(f"  [PASS] F1@20 = {f1_20:.2%} >= 70%, meets competition requirement!")
    else:
        gap_pct = (0.70 - f1_20) * 100
        print(f"  [GOAL] F1@20 = {f1_20:.2%}, gap to 70%: {gap_pct:.1f} pp")
        print(f"     PaSa-7B 在 RealScholarQuery 上的 F1 约为 ~0.45-0.50")
        print(f"     当前水平可作为 baseline，加入 Expand 后应有显著提升")

    # 与 PaSa 论文数据的对比
    print(f"\n{'─' * 40}")
    print("  [REF] PaSa paper reference (RealScholarQuery):")
    print("     PaSa-7B vs Google+GPT-4o: Recall@20 +37.78%, Recall@50 +39.90%")
    print("     PaSa-7B vs PaSa-GPT-4o:   Recall +30.36%, Precision +4.25%")

    return {
        "summary": avg_metrics,
        "per_query": per_query_results,
    }


def create_sample_dataset(output_path: str):
    """创建一个样例数据集，用于快速验证评测流程

    包含几条真实的研究查询，需要用户填写 Semantic Scholar 论文 ID
    """
    sample = [
        {
            "query": "retrieval augmented generation for knowledge-intensive NLP tasks",
            "relevant_papers": [
                "8e6de283a67253d23b5a1d2b7b561a98e93b0d5c",
            ],
        },
        {
            "query": "chain of thought prompting improves reasoning in large language models",
            "relevant_papers": [
                "1b6c1b3c9c4b3b3b3b3b3b3b3b3b3b3b3b3b3b3b",
            ],
        },
        {
            "query": "what are the recent advances in diffusion models for text-to-image generation",
            "relevant_papers": [],
        },
        {
            "query": "parameter efficient fine-tuning methods for large language models",
            "relevant_papers": [],
        },
        {
            "query": "how does RLHF improve instruction following in LLMs",
            "relevant_papers": [],
        },
    ]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[OK] Sample dataset created: {output_path}")
    print("   [WARN] Please replace placeholder paper IDs with actual Semantic Scholar IDs.")
    print(f"   Run: python evaluate.py {output_path} 5")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluate.py <dataset.jsonl>              # 评测全部")
        print("  python evaluate.py <dataset.jsonl> 10           # 评测前 10 条")
        print("  python evaluate.py --create-sample [output]     # 生成样例数据集")
        sys.exit(1)

    if sys.argv[1] == "--create-sample":
        output = sys.argv[2] if len(sys.argv) > 2 else "data/sample_queries.jsonl"
        create_sample_dataset(output)
    else:
        load_dotenv()

        # 检查 LLM 配置
        if not os.getenv("LLM_API_KEY") or not os.getenv("LLM_BASE_URL") or not os.getenv("LLM_MODEL"):
            print("[ERROR] Please configure LLM_API_KEY, LLM_BASE_URL, LLM_MODEL in .env")
            sys.exit(1)

        dataset_path = sys.argv[1]
        max_queries = int(sys.argv[2]) if len(sys.argv) > 2 else None

        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset file not found: {dataset_path}")
            sys.exit(1)

        results = run_evaluation(dataset_path, max_queries=max_queries)

        # 保存详细结果
        os.makedirs("output", exist_ok=True)
        output_file = f"output/eval_{int(time.time())}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] Detailed results saved to: {output_file}")
