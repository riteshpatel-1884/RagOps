"""Aggregates results into the 10 metrics that matter (with 95% bootstrap CIs) and compares versions."""
import json
from statistics import mean

from .config import GENERATOR_PRICE_PER_M, HIT_K, JUDGE_METRICS, UNANSWERABLE, VERSIONS, paths
from .metrics import hit_at_k, is_abstention
from .stats import latency_stats, summarize
from .text_utils import read_jsonl

JUDGE_LABELS = {
    "correctness": "Answer correctness", "faithfulness": "Faithfulness", "answer_relevancy": "Answer relevancy",
    "context_relevancy": "Context relevancy", "context_precision": "Context precision", "context_recall": "Context recall",
}


def _cost(row):
    return (row.get("input_tokens", 0) * GENERATOR_PRICE_PER_M["input"] + row.get("output_tokens", 0) * GENERATOR_PRICE_PER_M["output"]) / 1e6


def build(version: str) -> dict:
    p, spec = paths(version), VERSIONS[version]
    dataset = {r["id"]: r for r in read_jsonl(p["dataset"])}
    results = [r for r in read_jsonl(p["results"]) if not r.get("error")]
    answerable = [r for r in results if r["category"] != UNANSWERABLE]
    unans = [r for r in results if r["category"] == UNANSWERABLE]
    evaluated = read_jsonl(p["evaluated"])

    exp = {"version": version, "name": spec["name"], "spec": {k: spec[k] for k in ("chunking", "embedding", "retrieval")},
           "questions": {"total": len(results), "answerable": len(answerable), "unanswerable": len(unans)},
           "judged": len(evaluated)}

    exp["judge"] = {m: summarize([r.get(m) for r in evaluated]) for m in JUDGE_METRICS}

    hits = [(r, hit_at_k(r.get("retrieved_chunks"), dataset[r["id"]]["evidence"])) for r in answerable]
    exp["hit_rate"] = summarize([h for _, h in hits])
    exp["false_refusal_rate"] = summarize([int(is_abstention(r["actual_answer"])) for r in answerable])
    exp["hallucination_rate"] = summarize([int(not is_abstention(r["actual_answer"])) for r in unans])
    exp["abstention_accuracy"] = summarize([int(is_abstention(r["actual_answer"])) for r in unans])

    exp["latency"] = {k: latency_stats([r.get(k) for r in results]) for k in ("retrieval_ms", "ttft_ms", "total_ms")}
    exp["cost_per_query_usd"] = mean(_cost(r) for r in results) if results else None
    exp["tokens_per_query"] = {"input": mean(r.get("input_tokens", 0) for r in results) if results else None,
                               "output": mean(r.get("output_tokens", 0) for r in results) if results else None}

    by_cat = {}
    for cat in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == cat]
        d = {"n": len(rows)}
        if cat == UNANSWERABLE:
            d["abstention_accuracy"] = mean(int(is_abstention(r["actual_answer"])) for r in rows)
        else:
            d["hit_rate"] = mean(h for r, h in hits if r["category"] == cat)
            d["false_refusal_rate"] = mean(int(is_abstention(r["actual_answer"])) for r in rows)
            c = [e.get("correctness") for e in evaluated if e["category"] == cat and e.get("correctness") is not None]
            d["correctness"] = mean(c) if c else None
        by_cat[cat] = d
    exp["by_category"] = by_cat
    return exp


def _f(s, pct=False):
    if not s:
        return "n/a"
    v = f"{s['mean']:.3f}"
    return f"{v}  [{s['ci_low']:.3f} - {s['ci_high']:.3f}]"


def print_summary(exp):
    q, s = exp["questions"], exp["spec"]
    print(f"\n==================== {exp['name']} ====================")
    print(f"chunking: {s['chunking']['strategy']} | embedding: {s['embedding']['model']} ({s['embedding']['dim']}d) | "
          f"retrieval: {s['retrieval']['mode']} top_k={s['retrieval']['top_k']}")
    print(f"questions: {q['total']} ({q['answerable']} answerable, {q['unanswerable']} unanswerable) | judged: {exp['judged']}")
    print("\n--- Judge metrics (answerable questions): mean [95% CI] ---")
    for m in ("context_precision", "context_recall", "faithfulness", "answer_relevancy", "context_relevancy", "correctness"):
        print(f"{JUDGE_LABELS[m]:<20} {_f(exp['judge'][m])}   (n={(exp['judge'][m] or {}).get('n', 0)})")
    print(f"\n--- Retrieval and reliability (judge-free) ---")
    print(f"{'Hit rate @' + str(HIT_K):<20} {_f(exp['hit_rate'])}")
    print(f"{'False refusal rate':<20} {_f(exp['false_refusal_rate'])}   (answerable: wrongly refused)")
    print(f"{'Hallucination rate':<20} {_f(exp['hallucination_rate'])}   (unanswerable: answered anyway)")
    print(f"{'Abstention accuracy':<20} {_f(exp['abstention_accuracy'])}")
    print("\n--- Latency and cost ---")
    for k, label in (("retrieval_ms", "Retrieval"), ("ttft_ms", "TTFT"), ("total_ms", "Total")):
        v = exp["latency"][k]
        if v:
            print(f"{label:<20} p50 {v['p50']:.0f} ms | p95 {v['p95']:.0f} ms | mean {v['mean']:.0f} ms")
    if exp["cost_per_query_usd"] is not None:
        print(f"{'Cost per query':<20} ${exp['cost_per_query_usd']:.6f}   (${exp['cost_per_query_usd'] * 1000:.3f} per 1,000 queries)")
    print("\n--- By category ---")
    for cat, d in exp["by_category"].items():
        parts = [f"{k} {v:.2f}" for k, v in d.items() if k != "n" and v is not None]
        print(f"{cat:<14} n={d['n']:<3} | " + " | ".join(parts))


def save(version):
    exp = build(version)
    out = paths(version)["experiment"]
    out.write_text(json.dumps(exp, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Experiment saved to: {out}")


def compare():
    exps = []
    for v in VERSIONS:
        f = paths(v)["experiment"]
        if f.exists():
            exps.append(json.loads(f.read_text(encoding="utf-8")))
    if not exps:
        raise SystemExit("No experiment_vN.json found. Run `save <version>` first.")
    rows = [("Answer correctness", lambda e: (e["judge"]["correctness"] or {}).get("mean")),
            ("Faithfulness", lambda e: (e["judge"]["faithfulness"] or {}).get("mean")),
            ("Answer relevancy", lambda e: (e["judge"]["answer_relevancy"] or {}).get("mean")),
            ("Context relevancy", lambda e: (e["judge"]["context_relevancy"] or {}).get("mean")),
            ("Context precision", lambda e: (e["judge"]["context_precision"] or {}).get("mean")),
            ("Context recall", lambda e: (e["judge"]["context_recall"] or {}).get("mean")),
            (f"Hit rate @{HIT_K}", lambda e: (e["hit_rate"] or {}).get("mean")),
            ("False refusal rate", lambda e: (e["false_refusal_rate"] or {}).get("mean")),
            ("Hallucination rate", lambda e: (e["hallucination_rate"] or {}).get("mean")),
            ("TTFT p50 (ms)", lambda e: (e["latency"]["ttft_ms"] or {}).get("p50")),
            ("Total p50 (ms)", lambda e: (e["latency"]["total_ms"] or {}).get("p50")),
            ("Cost / query ($)", lambda e: e["cost_per_query_usd"])]
    print(f"\n{'Metric':<22}" + "".join(f"{e['version']:>12}" for e in exps))
    for label, get in rows:
        cells = []
        for e in exps:
            v = get(e)
            cells.append("n/a" if v is None else (f"{v:.0f}" if "ms" in label else (f"{v:.6f}" if "$" in label else f"{v:.3f}")))
        print(f"{label:<22}" + "".join(f"{c:>12}" for c in cells))
