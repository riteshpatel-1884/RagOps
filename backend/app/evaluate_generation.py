"""
Phase 2: Generation Evaluation Engine — main entry point.

For every test question: retrieve context (Phase 1) -> generate an answer ->
score Faithfulness and Answer Relevance.

Usage:
    python evaluate_generation.py
    python evaluate_generation.py --generator groq          # requires GROQ_API_KEY
    python evaluate_generation.py --chunk_size 300 --top_k 3
    python evaluate_generation.py --judge llm                # use LLM-as-judge instead of offline metrics
"""
import argparse
import json
from pathlib import Path

from app.pipeline import PipelineConfig, RetrievalPipeline
from app.gen_evaluators import (
    LexicalOverlapFaithfulness,
    EmbeddingSimilarityRelevance,
    LLMJudgeFaithfulness,
    LLMJudgeRelevance,
)

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    with open(DATA_DIR / "documents.json") as f:
        documents = json.load(f)
    with open(DATA_DIR / "test_dataset.json") as f:
        test_set = json.load(f)
    return documents, test_set


def build_evaluators(judge: str, pipeline: RetrievalPipeline):
    """judge: 'offline' (default, no LLM calls) or 'llm' (needs a real chat model)."""
    if judge == "offline":
        faithfulness_eval = LexicalOverlapFaithfulness()
        relevance_eval = EmbeddingSimilarityRelevance(pipeline.embedding)
    elif judge == "llm":
        if not hasattr(pipeline.generator, "chat_model"):
            raise ValueError(
                "--judge llm requires the groq generator, since it reuses that chat model as the judge."
            )
        chat_model = pipeline.generator.chat_model
        faithfulness_eval = LLMJudgeFaithfulness(chat_model)
        relevance_eval = LLMJudgeRelevance(chat_model)
    else:
        raise ValueError(f"Unknown judge mode: {judge}")
    return faithfulness_eval, relevance_eval


def run_generation_evaluation(config: PipelineConfig, judge: str = "offline", verbose: bool = True):
    documents, test_set = load_data()

    pipeline = RetrievalPipeline(config)
    pipeline.index(documents)

    faithfulness_eval, relevance_eval = build_evaluators(judge, pipeline)

    rows = []
    for item in test_set:
        result = pipeline.answer(item["question"])
        f_score = faithfulness_eval.score(result["answer"], result["context"])
        r_score = relevance_eval.score(item["question"], result["answer"])

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result["answer"],
                "faithfulness": f_score["faithfulness"],
                "faithfulness_explanation": f_score["explanation"],
                "relevance": r_score["relevance"],
                "relevance_explanation": r_score["explanation"],
            }
        )

    avg_faithfulness = sum(r["faithfulness"] for r in rows) / len(rows)
    avg_relevance = sum(r["relevance"] for r in rows) / len(rows)

    if verbose:
        print(f"\nPipeline config: {config.as_dict()}")
        print(f"Generator: {pipeline.generator.name}")
        print(f"Faithfulness evaluator: {faithfulness_eval.name}")
        print(f"Relevance evaluator: {relevance_eval.name}\n")

        for r in rows:
            print(f"[{r['id']}] {r['question']}")
            print(f"  Answer: {r['answer']}")
            print(f"  Faithfulness: {r['faithfulness']:.2f}  ({r['faithfulness_explanation']})")
            print(f"  Relevance:    {r['relevance']:.2f}  ({r['relevance_explanation']})")
            print()

        print("=== Overall Generation Metrics ===")
        print(f"  avg_faithfulness : {avg_faithfulness:.3f}")
        print(f"  avg_relevance    : {avg_relevance:.3f}")

    return {"avg_faithfulness": avg_faithfulness, "avg_relevance": avg_relevance, "rows": rows}


def parse_args():
    p = argparse.ArgumentParser(description="Run the Phase 2 Generation Evaluation Engine")
    p.add_argument("--chunk_size", type=int, default=400)
    p.add_argument("--chunk_overlap", type=int, default=80)
    p.add_argument(
        "--embedder", type=str, default="bge-small-en-v1.5",
        choices=["bge-m3", "qwen3-embedding-0.6b", "bge-small-en-v1.5"],
    )
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--generator", type=str, default="groq", choices=["groq"])
    p.add_argument(
        "--judge",
        type=str,
        default="offline",
        choices=["offline", "llm"],
        help="'offline' uses lexical-overlap + embedding-similarity (no LLM calls). "
        "'llm' reuses the groq generator's chat model as the judge.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedder_name=args.embedder,
        top_k=args.top_k,
        generator_name=args.generator,
    )
    run_generation_evaluation(config, judge=args.judge)