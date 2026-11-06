"""LLM-judge evaluation (DeepEval + Groq) with batching, resume, retry and per-metric saving.

Metrics: answer_relevancy, faithfulness, context_relevancy, correctness, completeness.
Only ANSWERABLE questions are judged (unanswerable ones are scored judge-free by abstention).

Examples (from backend/):
    python -m app.versions.v1.eval_all_metrics1                      # everything
    python -m app.versions.v1.eval_all_metrics1 --start 1 --end 20   # a slice (by question number)
    python -m app.versions.v1.eval_all_metrics1 --metrics correctness,completeness
    python -m app.versions.v1.eval_all_metrics1 --context-source llm # judge the LLM-written context instead

Safe to re-run: finished metrics are skipped, so it only spends tokens on what is missing.
"""
import argparse
import hashlib
import shutil
import time

from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_groq import ChatGroq

from .config import (DEFAULT_CONTEXT_SOURCE, JUDGE_METRICS, JUDGE_MODEL, JUDGE_THRESHOLD, UNANSWERABLE, paths)
from .migrate import archive_single_doc
from .retry import with_retry
from .text_utils import read_jsonl, write_jsonl

load_dotenv()


class GroqEvalModel(DeepEvalBaseLLM):

    def __init__(self, model_name=JUDGE_MODEL):
        self.model_name = model_name
        self.model = ChatGroq(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        if schema:
            return self.model.with_structured_output(schema, method="json_mode").invoke(prompt)
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        if schema:
            return await self.model.with_structured_output(schema, method="json_mode").ainvoke(prompt)
        return (await self.model.ainvoke(prompt)).content

    def get_model_name(self):
        return f"Groq {self.model_name}"


def make_metric(name, model):
    if name == "answer_relevancy":
        return AnswerRelevancyMetric(threshold=JUDGE_THRESHOLD, model=model)
    if name == "faithfulness":
        return FaithfulnessMetric(threshold=JUDGE_THRESHOLD, model=model)
    if name == "context_relevancy":
        return ContextualRelevancyMetric(threshold=JUDGE_THRESHOLD, model=model)
    if name == "correctness":
        return GEval(
            name="Correctness",
            criteria=("Determine whether the actual output is factually correct compared with the expected "
                      "output. Evaluate factual accuracy, missing information, and incorrect claims."),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=JUDGE_THRESHOLD, model=model)
    if name == "completeness":
        return GEval(
            name="Completeness",
            criteria=("Determine whether the actual output answers EVERY part of the input question and includes "
                      "all key facts of the expected output (dates, numbers, names, list items). Penalize missing "
                      "parts. Do not penalize extra detail that is correct."),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=JUDGE_THRESHOLD, model=model)
    raise ValueError(f"unknown metric {name}")


def retrieval_context(row, source):
    if source == "raw":
        return [c for c in (row.get("retrieved_chunks") or []) if c]
    ctx = row.get("context")
    return [ctx] if isinstance(ctx, str) and ctx else (ctx or [])


def row_key(row, source):
    """Changes whenever the answer or the context changes, so stale scores are never reused."""
    blob = "|".join([row["id"], row["question"], row.get("actual_answer") or "", source,
                     "\n".join(retrieval_context(row, source))])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def main(version: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="first question number (1-based, dataset order)")
    parser.add_argument("--end", type=int, default=None, help="last question number (inclusive)")
    parser.add_argument("--metrics", default=",".join(JUDGE_METRICS), help="comma separated subset")
    parser.add_argument("--context-source", choices=["raw", "llm"], default=DEFAULT_CONTEXT_SOURCE)
    parser.add_argument("--sleep", type=float, default=15, help="seconds between questions")
    parser.add_argument("--judge-model", default=JUDGE_MODEL,
                        help="override ONLY if you re-judge every version with the same model")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in JUDGE_METRICS]
    if bad:
        raise SystemExit(f"Unknown metrics {bad}. Choose from {JUDGE_METRICS}")

    archive_single_doc(version)
    p = paths(version)
    results = read_jsonl(p["results"])
    if not results:
        raise SystemExit(f"{p['results'].name} is missing or empty. Run the eval_runner for {version} first.")

    old = read_jsonl(p["evaluated"])
    if old and any("key" not in r for r in old):
        legacy = p["evaluated"].with_name(p["evaluated"].stem + ".legacy.jsonl")
        if not legacy.exists():
            shutil.copy(p["evaluated"], legacy)
            print(f"Old-format evaluation backed up to {legacy.name}")

    evaluated = {r["key"]: r for r in old if "key" in r}
    source = args.context_source
    model = GroqEvalModel(args.judge_model)

    def save():
        keep = []
        for r in results:
            if r.get("category") == UNANSWERABLE or r.get("error") or not r.get("actual_answer"):
                continue
            k = row_key(r, source)
            if k in evaluated:
                keep.append(evaluated[k])
        write_jsonl(p["evaluated"], keep)

    end = min(args.end or len(results), len(results))
    print(f"{version}: judging questions {args.start}-{end} of {len(results)} | judge {args.judge_model} | "
          f"context source: {source} | metrics: {', '.join(metrics)}")

    for i in range(args.start, end + 1):
        r = results[i - 1]
        if r.get("category") == UNANSWERABLE:
            print(f"\n[{i}/{len(results)}] {r['id']}: unanswerable, scored by the abstention check (no judge needed)")
            continue
        if r.get("error") or not r.get("actual_answer"):
            print(f"\n[{i}] {r['id']}: no answer (error), skipping")
            continue

        key = row_key(r, source)
        row = evaluated.get(key) or {
            "key": key, "id": r["id"], "category": r["category"], "source": r.get("source"), "question": r["question"],
            "expected_answer": r["expected_answer"], "actual_answer": r["actual_answer"],
            "context_source": source, "judge_model": args.judge_model,
        }
        evaluated[key] = row

        pending = [m for m in metrics if row.get(m) is None]
        if not pending:
            print(f"\n[{i}/{len(results)}] {r['id']}: already judged, skipping")
            continue

        print(f"\n[{i}/{len(results)}] {r['id']} ({r['category']}): judging {', '.join(pending)}")
        ctx = retrieval_context(r, source)
        test_case = LLMTestCase(input=r["question"], actual_output=r["actual_answer"],
                                expected_output=r["expected_answer"], retrieval_context=ctx)

        for m in pending:
            metric = make_metric(m, model)
            with_retry(lambda: metric.measure(test_case), m)
            row[m] = metric.score
            row[f"{m}_passed"] = bool(metric.is_successful())
            print(f"  {m}: {metric.score}")
            save()                                    # saved after EVERY metric

        time.sleep(args.sleep)

    save()
    judged = [r for r in results if r.get("category") != UNANSWERABLE and not r.get("error") and r.get("actual_answer")]
    full = sum(1 for r in judged if all((evaluated.get(row_key(r, source)) or {}).get(m) is not None for m in JUDGE_METRICS))
    print(f"\nFully judged (all {len(JUDGE_METRICS)} metrics): {full}/{len(judged)} answerable questions")
    print(f"Saved: {p['evaluated']}")
    if full == len(judged):
        print("All done. Next: evaluation_summary and save_experiment.")
    else:
        print("Some metrics are still pending. Re-run the same command (or the next batch).")
