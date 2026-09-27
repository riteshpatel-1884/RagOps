"""
Phase 3: Experiment Engine — main entry point.

This is what turns the project from a single-run evaluation script into an
*optimization* engine: define a grid of pipeline configs (chunk_size,
chunk_overlap, embedder, top_k, generator), run every combination against
the same test set, score each with Phase 1 (retrieval) + Phase 2
(generation) metrics, and rank them.

Usage:
    # Fast retrieval-only sweep (no LLM calls, good for quick iteration)
    python experiment.py --chunk_sizes 200 400 --top_ks 3 5 --skip_generation

    # Full sweep including generation metrics
    python experiment.py --chunk_sizes 200 400 --chunk_overlaps 40 80 --top_ks 3 5

    # Compare embedding models
    python experiment.py --embedders bge-small-en-v1.5 bge-m3 --top_ks 5

    # Rank by a specific metric instead of the default (recall)
    python experiment.py --chunk_sizes 200 400 --sort_by faithfulness
"""
import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

from app.pipeline import PipelineConfig, RetrievalPipeline
from app.metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, aggregate_metrics
from app.gen_evaluators import (
    LexicalOverlapFaithfulness,
    EmbeddingSimilarityRelevance,
    LLMJudgeFaithfulness,
    LLMJudgeRelevance,
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "experiment_results"


def load_data():
    with open(DATA_DIR / "documents.json") as f:
        documents = json.load(f)
    with open(DATA_DIR / "test_dataset.json") as f:
        test_set = json.load(f)
    return documents, test_set


def build_gen_evaluators(judge: str, pipeline: RetrievalPipeline):
    if judge == "offline":
        return LexicalOverlapFaithfulness(), EmbeddingSimilarityRelevance(pipeline.embedding)
    if judge == "llm":
        if not hasattr(pipeline.generator, "chat_model"):
            raise ValueError(
                "judge='llm' requires the groq generator, since it reuses that chat model as the judge."
            )
        chat_model = pipeline.generator.chat_model
        return LLMJudgeFaithfulness(chat_model), LLMJudgeRelevance(chat_model)
    raise ValueError(f"Unknown judge mode: {judge}")


def run_single_config(config: PipelineConfig, documents: dict, test_set: list, judge: str, skip_generation: bool) -> dict:
    """Run retrieval (+ optionally generation) evaluation for one pipeline config."""
    start = time.time()

    pipeline = RetrievalPipeline(config)
    pipeline.index(documents)

    faithfulness_eval, relevance_eval = (None, None)
    if not skip_generation:
        faithfulness_eval, relevance_eval = build_gen_evaluators(judge, pipeline)

    retrieval_rows, gen_rows = [], []

    for item in test_set:
        relevant_ids = pipeline.chunk_ids_for_docs(item["relevant_doc_ids"])
        # Fetch a bit beyond top_k so recall/precision at top_k are still exact
        # even if the store's default limit were smaller.
        hits = pipeline.retrieve(item["question"], top_k=max(config.top_k, 10))
        retrieved_ids = [h["chunk_id"] for h in hits]

        retrieval_rows.append(
            {
                "recall": recall_at_k(retrieved_ids, relevant_ids, config.top_k),
                "precision": precision_at_k(retrieved_ids, relevant_ids, config.top_k),
                "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
                "ndcg": ndcg_at_k(retrieved_ids, relevant_ids, config.top_k),
            }
        )

        if not skip_generation:
            top_hits = hits[: config.top_k]
            context = [h["text"] for h in top_hits]
            answer = pipeline.generator.generate(item["question"], context)
            f_score = faithfulness_eval.score(answer, context)
            r_score = relevance_eval.score(item["question"], answer)
            gen_rows.append({"faithfulness": f_score["faithfulness"], "relevance": r_score["relevance"]})

    elapsed = time.time() - start

    result = {"config": config.as_dict(), **aggregate_metrics(retrieval_rows), "elapsed_seconds": round(elapsed, 2)}
    if gen_rows:
        result.update(aggregate_metrics(gen_rows))

    return result


def build_grid(args) -> List[PipelineConfig]:
    """Cartesian product of every swept dimension, skipping invalid combos."""
    configs = []
    for cs, co, emb, k, gen in itertools.product(
        args.chunk_sizes, args.chunk_overlaps, args.embedders, args.top_ks, args.generators
    ):
        if co >= cs:
            continue  # overlap must be smaller than chunk size
        configs.append(
            PipelineConfig(chunk_size=cs, chunk_overlap=co, embedder_name=emb, top_k=k, generator_name=gen)
        )
    return configs


def rank_and_print(results: List[dict], sort_by: str) -> List[dict]:
    results = sorted(results, key=lambda r: r.get(sort_by, 0), reverse=True)
    metric_keys = [k for k in results[0].keys() if k != "config"]
    widths = {m: max(len(m) + 2, 10) for m in metric_keys}

    header = f"{'Rank':<5}{'chunk':<7}{'ovlp':<6}{'embedder':<12}{'top_k':<7}{'generator':<12}"
    header += "".join(f"{m:<{widths[m]}}" for m in metric_keys)
    print("\n" + header)
    print("-" * len(header))

    for i, r in enumerate(results, start=1):
        c = r["config"]
        row = f"{i:<5}{c['chunk_size']:<7}{c['chunk_overlap']:<6}{c['embedder_name']:<12}{c['top_k']:<7}{c['generator_name']:<12}"
        for m in metric_keys:
            val = r[m]
            row += f"{val:<{widths[m]}.3f}" if isinstance(val, float) else f"{val:<{widths[m]}}"
        print(row)

    print(f"\nRanked by: {sort_by} (descending)")
    print(f"Best config: {results[0]['config']}")
    return results


def save_results(results: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: sweep pipeline configs and rank them")
    p.add_argument("--chunk_sizes", type=int, nargs="+", default=[200, 400])
    p.add_argument("--chunk_overlaps", type=int, nargs="+", default=[40, 80])
    p.add_argument(
        "--embedders", type=str, nargs="+", default=["bge-small-en-v1.5"],
        choices=["bge-m3", "qwen3-embedding-0.6b", "bge-small-en-v1.5"],
    )
    p.add_argument("--top_ks", type=int, nargs="+", default=[3, 5])
    p.add_argument("--generators", type=str, nargs="+", default=["groq"], choices=["groq"])
    p.add_argument("--judge", type=str, default="offline", choices=["offline", "llm"])
    p.add_argument(
        "--skip_generation",
        action="store_true",
        help="Only run retrieval metrics — much faster for large sweeps, no generator/LLM calls at all.",
    )
    p.add_argument(
        "--sort_by",
        type=str,
        default="recall",
        help="Metric to rank configs by: recall, precision, mrr, ndcg, faithfulness, relevance, or elapsed_seconds.",
    )
    return p.parse_args()


def run_sweep(configs: List[PipelineConfig], documents: dict, test_set: list, judge: str = "offline", skip_generation: bool = False) -> List[dict]:
    """
    Run every config in the grid and collect results. Used by both the CLI
    (main(), below) and the API server, so the sweep logic only lives once.
    Failed configs are included in the output with an "error" key rather
    than being silently dropped.
    """
    results = []
    for config in configs:
        try:
            results.append(run_single_config(config, documents, test_set, judge, skip_generation))
        except Exception as e:
            results.append({"config": config.as_dict(), "error": str(e)})
    return results


def main():
    args = parse_args()
    documents, test_set = load_data()
    configs = build_grid(args)

    if not configs:
        print("No valid configs generated — check that every chunk_overlap is smaller than every chunk_size.")
        return

    print(f"Running {len(configs)} pipeline configuration(s) against {len(test_set)} test questions...\n")

    all_results = run_sweep(configs, documents, test_set, args.judge, args.skip_generation)
    for i, r in enumerate(all_results, start=1):
        if "error" in r:
            print(f"[{i}/{len(all_results)}] {r['config']}\n  FAILED: {r['error']}")
        else:
            print(f"[{i}/{len(all_results)}] {r['config']}")

    results = [r for r in all_results if "error" not in r]
    if not results:
        print("\nAll configs failed — see errors above.")
        return

    sort_by = args.sort_by if args.sort_by in results[0] else "recall"
    ranked = rank_and_print(results, sort_by)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(ranked, RESULTS_DIR / f"experiment_{timestamp}.json")


if __name__ == "__main__":
    main()