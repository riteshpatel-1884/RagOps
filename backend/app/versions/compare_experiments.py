# Compare experiments side by side.
# Usage (from backend/):
#   python -m app.versions.compare_experiments              # every version with an experiment file, baseline = first
#   python -m app.versions.compare_experiments v1 v3        # only these (first one is the baseline)
import json
import re
import sys
from pathlib import Path

versions_dir = Path(__file__).resolve().parent


def load(version):
    path = versions_dir / version / f"experiment_{version}.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run save_experiment for {version} first.")
    return json.loads(path.read_text(encoding="utf-8"))


if len(sys.argv) > 1:
    versions = sys.argv[1:]
else:
    versions = sorted(
        (p.name for p in versions_dir.iterdir()
         if p.is_dir() and re.fullmatch(r"v\d+", p.name) and (p / f"experiment_{p.name}.json").exists()),
        key=lambda v: int(v[1:]))

if len(versions) < 2:
    raise SystemExit("Need at least 2 versions with experiment files to compare.")

exps = {v: load(v) for v in versions}
baseline = versions[0]
W = 24


def dig(exp, path):
    cur = exp
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def mean_of(x):
    return x["mean"] if isinstance(x, dict) and "mean" in x else x


def judge_value(exp, m):
    """New-format experiments have judge.<m>; old ones only metrics.<m>."""
    v = dig(exp, f"judge.{m}")
    return v if v is not None else (dig(exp, f"metrics.{m}") if m != "completeness" else None)


def fmt(v, kind):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, dict) and "mean" in v:
        s = f"{v['mean']:.2f}" if kind != "ms" else f"{v['mean']:.0f}"
        if kind == "ci" and v.get("ci_low") is not None:
            return f"{v['mean']:.3f} [{v['ci_low']:.2f}-{v['ci_high']:.2f}]"
        return s
    if kind == "ms":
        return f"{v:.0f}"
    if kind == "usd":
        return f"${v:.5f}"
    if kind == "usd3":
        return f"${v:.3f}"
    if kind == "int":
        return f"{v:.0f}"
    return f"{v:.3f}"


def table(title, rows):
    print(f"\n---------- {title} ----------")
    print(f"{'':<34}" + "".join(f"{v:<{W}}" for v in versions))
    for label, getter, kind in rows:
        vals = [fmt(getter(exps[v]), kind) for v in versions]
        if all(x == "n/a" for x in vals):
            continue
        print(f"{label:<34}" + "".join(f"{x:<{W}}" for x in vals))


print("\n=============== EXPERIMENT COMPARISON ===============")
print(f"{'':<34}" + "".join(f"{v:<{W}}" for v in versions))
print(f"{'retriever k':<34}" + "".join(f"{'k=' + str(exps[v].get('retriever_k')):<{W}}" for v in versions))
print(f"{'vector DB':<34}" + "".join(f"{str(dig(exps[v], 'corpus.vector_db') or 'n/a'):<{W}}" for v in versions))
print(f"{'documents':<34}" + "".join(f"{str(len(dig(exps[v], 'corpus.documents') or [])) or 'n/a':<{W}}" for v in versions))
print(f"{'chunking':<34}" + "".join(f"{str(exps[v].get('chunking', 'n/a'))[:W - 2]:<{W}}" for v in versions))
print(f"{'questions judged':<34}" + "".join(f"{str(dig(exps[v], 'dataset.judged') or exps[v].get('number_of_test_cases', 'n/a')):<{W}}" for v in versions))

table("Judge metrics, answerable questions: mean [95% CI]", [
    ("Answer relevancy", lambda e: judge_value(e, "answer_relevancy"), "ci"),
    ("Faithfulness", lambda e: judge_value(e, "faithfulness"), "ci"),
    ("Context relevancy", lambda e: judge_value(e, "context_relevancy"), "ci"),
    ("Correctness", lambda e: judge_value(e, "correctness"), "ci"),
    ("Completeness", lambda e: judge_value(e, "completeness"), "ci"),
])

table("Retrieval at each version's own k", [
    ("Hit rate", lambda e: dig(e, "retrieval.at_k.hit"), "ci"),
    ("Recall", lambda e: dig(e, "retrieval.at_k.fact_recall"), "ci"),
    ("Precision", lambda e: dig(e, "retrieval.at_k.precision"), "ci"),
    ("MRR", lambda e: dig(e, "retrieval.at_k.mrr"), "ci"),
    ("Right-document hit", lambda e: dig(e, "retrieval.at_k.source_hit"), "ci"),
    ("All needed docs reached", lambda e: dig(e, "retrieval.at_k.source_recall"), "ci"),
    ("Right-document precision", lambda e: dig(e, "retrieval.at_k.source_precision"), "ci"),
])

table("Retrieval curve at fixed depths (hit / recall)", [
    (f"k={k}", (lambda kk: lambda e: (
        f"{dig(e, f'retrieval.curve.{kk}.hit'):.2f} / {dig(e, f'retrieval.curve.{kk}.fact_recall'):.2f}"
        if dig(e, f"retrieval.curve.{kk}.hit") is not None else None))(k), "str")
    for k in (1, 3, 5)
])

table("Generation (judge-free)", [
    ("Key-fact coverage", lambda e: dig(e, "generation.fact_coverage"), "ci"),
    ("Abstention accuracy", lambda e: dig(e, "generation.abstention_accuracy"), "ci"),
    ("Hallucination rate (lower=better)", lambda e: dig(e, "generation.hallucination_rate"), "ci"),
    ("False refusal (lower=better)", lambda e: dig(e, "generation.false_refusal_rate"), "ci"),
    ("Context grounding", lambda e: dig(e, "generation.context_grounding"), "ci"),
    ("Paraphrase consistency", lambda e: dig(e, "generation.paraphrase_consistency"), "score"),
])

table("Performance", [
    ("Retrieval latency p50 (ms)", lambda e: dig(e, "performance.latency_ms.retrieval_ms.p50"), "ms"),
    ("Generation latency p50 (ms)", lambda e: dig(e, "performance.latency_ms.generation_ms.p50"), "ms"),
    ("Total latency p50 (ms)", lambda e: dig(e, "performance.latency_ms.total_ms.p50"), "ms"),
    ("Total latency p95 (ms)", lambda e: dig(e, "performance.latency_ms.total_ms.p95"), "ms"),
    ("Input tokens / query", lambda e: dig(e, "performance.tokens_per_query.input"), "int"),
    ("Output tokens / query", lambda e: dig(e, "performance.tokens_per_query.output"), "int"),
    ("Cost / 1,000 queries", lambda e: dig(e, "performance.cost_per_1k_queries_usd"), "usd3"),
])

cats = sorted({c for e in exps.values() for c in (e.get("by_category") or {})})
if cats:
    table("Correctness by question category", [
        (c, (lambda cc: lambda e: dig(e, f"by_category.{cc}.correctness"))(c), "score")
        for c in cats if c != "unanswerable"])

cats_hit = sorted({c for e in exps.values() for c in (e.get("by_category") or {})})
table("Retrieval hit rate by category (own k)", [
    (c, (lambda cc: lambda e: dig(e, f"by_category.{cc}.hit"))(c), "score")
    for c in cats_hit if c != "unanswerable"])

srcs = sorted({s_ for e in exps.values() for s_ in (e.get("by_source") or {})})
if srcs:
    table("Correctness by source document", [
        (s_[:32], (lambda ss: lambda e: ((e.get("by_source") or {}).get(ss) or {}).get("correctness"))(s_), "score")
        for s_ in srcs])
    table("Retrieval hit rate by source document", [
        (s_[:32], (lambda ss: lambda e: ((e.get("by_source") or {}).get(ss) or {}).get("hit"))(s_), "score")
        for s_ in srcs])

# change vs baseline on the headline numbers (means only)
print(f"\n---------- Change vs baseline ({baseline}) ----------")
others = versions[1:]
print(f"{'':<34}" + "".join(f"{v:<{W}}" for v in others))
headline = [
    ("Correctness", lambda e: mean_of(judge_value(e, "correctness"))),
    ("Completeness", lambda e: mean_of(judge_value(e, "completeness"))),
    ("Faithfulness", lambda e: mean_of(judge_value(e, "faithfulness"))),
    ("Context relevancy", lambda e: mean_of(judge_value(e, "context_relevancy"))),
    ("Retrieval hit rate", lambda e: mean_of(dig(e, "retrieval.at_k.hit"))),
    ("Retrieval MRR", lambda e: mean_of(dig(e, "retrieval.at_k.mrr"))),
    ("Right-document hit", lambda e: mean_of(dig(e, "retrieval.at_k.source_hit"))),
    ("Key-fact coverage", lambda e: mean_of(dig(e, "generation.fact_coverage"))),
    ("Hallucination rate", lambda e: mean_of(dig(e, "generation.hallucination_rate"))),
]
for label, g in headline:
    b = g(exps[baseline])
    cells = []
    for v in others:
        x = g(exps[v])
        cells.append("n/a" if b is None or x is None else f"{x - b:+.3f}")
    if any(c != "n/a" for c in cells):
        print(f"{label:<34}" + "".join(f"{c:<{W}}" for c in cells))

print("\nCI = 95% bootstrap interval. Overlapping intervals mean the difference may be noise.\n")
