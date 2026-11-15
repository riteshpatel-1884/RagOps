"""LLM-as-a-judge with DeepEval (Groq judge): answer relevancy, faithfulness, context relevancy,
context precision, context recall, correctness. Only ANSWERABLE questions are judged.

Safe to re-run: finished metrics are skipped, progress is saved after every metric.
"""
import hashlib
import time

from deepeval.metrics import (AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric,
                              ContextualRelevancyMetric, FaithfulnessMetric, GEval)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from .config import JUDGE_METRICS, JUDGE_MODEL, JUDGE_THRESHOLD, UNANSWERABLE, paths
from .retry import with_retry
from .text_utils import read_jsonl, write_jsonl


class GroqEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name=JUDGE_MODEL):
        from langchain_groq import ChatGroq
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
    t = JUDGE_THRESHOLD
    if name == "answer_relevancy":
        return AnswerRelevancyMetric(threshold=t, model=model)
    if name == "faithfulness":
        return FaithfulnessMetric(threshold=t, model=model)
    if name == "context_relevancy":
        return ContextualRelevancyMetric(threshold=t, model=model)
    if name == "context_precision":
        return ContextualPrecisionMetric(threshold=t, model=model)
    if name == "context_recall":
        return ContextualRecallMetric(threshold=t, model=model)
    if name == "correctness":
        return GEval(name="Correctness",
                     criteria=("Determine whether the actual output is factually correct compared with the expected "
                               "output. Evaluate factual accuracy, missing information, and incorrect claims."),
                     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                     threshold=t, model=model)
    raise ValueError(f"unknown metric {name}")


def row_key(row):
    """Changes whenever the answer or the retrieved context changes, so stale scores are never reused."""
    blob = "|".join([row["id"], row["question"], row.get("actual_answer") or "", "\n".join(row.get("retrieved_chunks") or [])])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def run_judge(version, start=1, end=None, metrics=None, sleep=10, every=1):
    metrics = metrics or JUDGE_METRICS
    p = paths(version)
    results = read_jsonl(p["results"])
    if not results:
        raise SystemExit(f"{p['results'].name} is missing. Run:  python -m app.versions.cli run {version}")

    evaluated = {r["key"]: r for r in read_jsonl(p["evaluated"]) if "key" in r}
    model = GroqEvalModel()

    def judged(r):
        return r.get("category") != UNANSWERABLE and not r.get("error") and r.get("actual_answer")

    def save():
        write_jsonl(p["evaluated"], [evaluated[row_key(r)] for r in results if judged(r) and row_key(r) in evaluated])

    end = min(end or len(results), len(results))
    print(f"{version}: judging questions {start}-{end} of {len(results)} | judge {JUDGE_MODEL} | metrics: {', '.join(metrics)}")

    for i in range(start, end + 1):
        r = results[i - 1]
        if not judged(r):
            print(f"[{i}/{len(results)}] {r['id']}: {'unanswerable, no judge needed' if r.get('category') == UNANSWERABLE else 'no answer, skipped'}")
            continue
        if every > 1 and (i - 1) % every != 0:
            continue                              # judge only every Nth question (same subset for every version)
        key = row_key(r)
        row = evaluated.setdefault(key, {"key": key, "id": r["id"], "category": r["category"], "question": r["question"],
                                         "expected_answer": r["expected_answer"], "actual_answer": r["actual_answer"]})
        pending = [m for m in metrics if row.get(m) is None]
        if not pending:
            print(f"[{i}/{len(results)}] {r['id']}: already judged")
            continue
        print(f"[{i}/{len(results)}] {r['id']} ({r['category']}): judging {', '.join(pending)}")
        case = LLMTestCase(input=r["question"], actual_output=r["actual_answer"], expected_output=r["expected_answer"],
                           retrieval_context=[c for c in r.get("retrieved_chunks") or [] if c])
        for m in pending:
            metric = make_metric(m, model)
            with_retry(lambda: metric.measure(case), m)
            row[m] = metric.score
            print(f"  {m}: {metric.score}")
            save()
        time.sleep(sleep)

    save()
    todo = [r for i, r in enumerate(results) if judged(r) and (every <= 1 or i % every == 0)]
    full = sum(1 for r in todo if all((evaluated.get(row_key(r)) or {}).get(m) is not None for m in JUDGE_METRICS))
    print(f"\nFully judged: {full}/{len(todo)} answerable questions -> {p['evaluated']}")
    print("All done. Next: summary / save." if full == len(todo) else "Some metrics are pending: re-run the same command (or the next batch).")
