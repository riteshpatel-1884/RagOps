"""
Phase 1: Retrieval Evaluation Engine — main entry point.

Usage:
    python evaluate.py
    python evaluate.py --chunk_size 200 --chunk_overlap 40 --top_k 3
    python evaluate.py --embedder bge-m3               # BAAI/bge-m3
    python evaluate.py --embedder qwen3-embedding-0.6b  # Qwen/Qwen3-Embedding-0.6B

Loads documents + a labeled test dataset, builds a retrieval pipeline with
the given config, runs every test question through it, and prints
Recall@K / Precision@K / MRR / NDCG@K — both per-query and averaged.
"""
import argparse
import json
from pathlib import Path

from pipeline import PipelineConfig, RetrievalPipeline
from metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, first_relevant_rank, aggregate_metrics

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    with open(DATA_DIR / "documents.json") as f:
        documents = json.load(f)
    with open(DATA_DIR / "test_dataset.json") as f:
        test_set = json.load(f)
    return documents, test_set


def run_evaluation(config: PipelineConfig, verbose: bool = True):
    documents, test_set = load_data()

    pipeline = RetrievalPipeline(config)
    pipeline.index(documents)

    per_query_metrics = []
    per_query_report = []

    for item in test_set:
        question = item["question"]
        relevant_ids = pipeline.chunk_ids_for_docs(item["relevant_doc_ids"])

        hits = pipeline.retrieve(question, top_k=max(config.top_k, 10))
        retrieved_ids = [h["chunk_id"] for h in hits]

        m = {
            f"recall@{config.top_k}": recall_at_k(retrieved_ids, relevant_ids, config.top_k),
            f"precision@{config.top_k}": precision_at_k(retrieved_ids, relevant_ids, config.top_k),
            "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
            f"ndcg@{config.top_k}": ndcg_at_k(retrieved_ids, relevant_ids, config.top_k),
        }
        per_query_metrics.append(m)
        per_query_report.append(
            {
                "id": item["id"],
                "question": question,
                "top_hit_doc": hits[0]["doc_id"] if hits else None,
                "expected_docs": item["relevant_doc_ids"],
                "first_relevant_rank": first_relevant_rank(retrieved_ids, relevant_ids),
                **m,
            }
        )

    overall = aggregate_metrics(per_query_metrics)

    if verbose:
        print(f"\nPipeline config: {config.as_dict()}")
        print(f"Embedder in use: {getattr(pipeline.embedding, 'name', type(pipeline.embedding).__name__)}")
        print(f"Indexed {len(pipeline.chunks)} chunks from {len(documents)} documents\n")

        print(f"{'ID':<4} {'1st rel. rank':<14} {'Recall@K':<10} {'Prec@K':<10} {'MRR':<8} Question")
        print("-" * 100)
        for r in per_query_report:
            rank_str = str(r["first_relevant_rank"]) if r["first_relevant_rank"] != -1 else "not found"
            print(
                f"{r['id']:<4} {rank_str:<14} "
                f"{r[f'recall@{config.top_k}']:<10.2f} {r[f'precision@{config.top_k}']:<10.2f} "
                f"{r['mrr']:<8.2f} {r['question'][:55]}"
            )

        print("\n=== Overall Metrics (averaged across all queries) ===")
        for k, v in overall.items():
            print(f"  {k:<15}: {v:.3f}")

    return overall, per_query_report


def parse_args():
    p = argparse.ArgumentParser(description="Run the Phase 1 Retrieval Evaluation Engine")
    p.add_argument("--chunk_size", type=int, default=400, help="characters")
    p.add_argument("--chunk_overlap", type=int, default=80, help="characters")
    p.add_argument(
        "--embedder", type=str, default="bge-small-en-v1.5",
        choices=["bge-m3", "qwen3-embedding-0.6b", "bge-small-en-v1.5"],
    )
    p.add_argument("--top_k", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedder_name=args.embedder,
        top_k=args.top_k,
    )
    run_evaluation(config)