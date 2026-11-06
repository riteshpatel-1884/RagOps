"""Build the full experiment report for one version from its three result files."""
from collections import defaultdict
from statistics import mean

from .config import (GENERATOR_PRICE_PER_M, JUDGE_METRICS, JUDGE_MODEL, JUDGE_THRESHOLD, RETRIEVAL_DEPTHS,
                     UNANSWERABLE, VERSIONS, paths)
from .generation_metrics import fact_coverage, is_abstention, ngram_grounding
from .stats import latency_stats, summarize
from .text_utils import read_jsonl

RETRIEVAL_KEYS = ("hit", "fact_recall", "precision", "mrr", "source_hit", "source_recall", "source_precision")
LEGACY_KEYS = ("answer_relevancy", "faithfulness", "context_relevancy", "correctness")


def _consistency(groups):
    """groups: {name: [values]} -> mean of (1 - spread) over groups with >= 2 values (1.0 = identical)."""
    spreads = []
    for vals in groups.values():
        vals = [x for x in vals if x is not None]
        if len(vals) >= 2:
            spreads.append(1 - (max(vals) - min(vals)))
    return mean(spreads) if spreads else None


def _src(x):
    """Dataset `source` (list of PDF names) as one readable key, e.g. 'ritesh_AI.pdf + NEXORA_TECHNOLOGIES_PVT.pdf'."""
    if isinstance(x, str):
        return x
    return " + ".join(sorted(x)) if x else "unknown"


def build_report(version: str) -> dict:
    cfg, p = VERSIONS[version], paths(version)
    dataset = read_jsonl(p["dataset"])
    # rows written by the old (pre-toolkit) scripts have no category/key: ignore them
    results = [r for r in read_jsonl(p["results"]) if "category" in r and "retrieved_sources" in r]
    evaluated = [e for e in read_jsonl(p["evaluated"]) if "key" in e and "source" in e]
    retrieval = [r for r in read_jsonl(p["retrieval"]) if "at" in r and "retrieved_sources" in r]

    ok = [r for r in results if r.get("error") is None and r.get("actual_answer") is not None]
    answerable = [r for r in ok if r["category"] != UNANSWERABLE]
    unanswerable = [r for r in ok if r["category"] == UNANSWERABLE]

    k = next((r["retriever_k"] for r in ok + retrieval if r.get("retriever_k")), None)
    report = {
        "experiment": cfg["name"], "version": version, "retriever_k": k, "chunking": cfg["chunking"],
        "corpus": {"vector_db": "Qdrant", "documents": sorted({d for r in dataset for d in (r.get("source") or [])})},
        "judge_model": next((e.get("judge_model") for e in evaluated if e.get("judge_model")), JUDGE_MODEL),
        "context_source": next((e.get("context_source") for e in evaluated if e.get("context_source")), None),
        "dataset": {
            "questions_in_dataset": len(dataset), "answers_generated": len(ok),
            "generation_errors": len(results) - len(ok),
            "answerable": len(answerable), "unanswerable": len(unanswerable),
            "judged": len(evaluated),
            "by_category": {c: sum(1 for r in dataset if r["category"] == c) for c in sorted({r["category"] for r in dataset})},
        },
    }

    # ---------------- retrieval (judge-free) ----------------
    if retrieval:
        depths = sorted({int(d) for r in retrieval for d in r["at"]})
        def agg(depth):
            return {m: summarize([r["at"][str(depth)].get(m) for r in retrieval]) for m in RETRIEVAL_KEYS}
        report["retrieval"] = {
            "at_k": agg(k) if k else None,
            "curve": {str(d): {m: (v["mean"] if v else None) for m, v in agg(d).items()} for d in depths},
            "latency_ms": latency_stats([r["retrieval_ms"] for r in retrieval]),
            "misses": [r["id"] for r in retrieval if r["at"][str(max(depths))]["hit"] == 0],
        }
    else:
        report["retrieval"] = None

    # ---------------- generation (judge-free) ----------------
    gen = {}
    per = []
    for r in ok:
        cov = fact_coverage(r["actual_answer"], r.get("gold_facts"))
        per.append({"id": r["id"], "category": r["category"], "group": r.get("paraphrase_group"),
                    "abstained": is_abstention(r["actual_answer"]), "coverage": cov,
                    "grounding": ngram_grounding(r.get("context"), r.get("retrieved_chunks") or [])
                    if r.get("retrieved_chunks") else None})
    ans = [x for x in per if x["category"] != UNANSWERABLE]
    una = [x for x in per if x["category"] == UNANSWERABLE]
    gen["fact_coverage"] = summarize([x["coverage"] for x in ans])
    gen["abstention_accuracy"] = summarize([float(x["abstained"]) for x in una])      # unanswerable: should refuse
    gen["hallucination_rate"] = summarize([float(not x["abstained"]) for x in una])    # unanswerable: answered anyway
    gen["false_refusal_rate"] = summarize([float(x["abstained"] and (x["coverage"] or 0) < 0.5) for x in ans])
    gen["context_grounding"] = summarize([x["grounding"] for x in per])
    groups = defaultdict(list)
    for x in ans:
        if x["group"]:
            groups[x["group"]].append(x["coverage"])
    gen["paraphrase_consistency"] = _consistency(groups)
    rgroups = defaultdict(list)
    for r in retrieval:
        if r.get("paraphrase_group") and k:
            rgroups[r["paraphrase_group"]].append(r["at"][str(k)]["fact_recall"])
    gen["paraphrase_retrieval_consistency"] = _consistency(rgroups)
    report["generation"] = gen

    # ---------------- judge metrics ----------------
    judge = {}
    for m in JUDGE_METRICS:
        vals = [e.get(m) for e in evaluated if e.get(m) is not None]
        if vals:
            s = summarize(vals)
            s["pass_rate"] = sum(v >= JUDGE_THRESHOLD for v in vals) / len(vals)
            judge[m] = s
    report["judge"] = judge
    report["metrics"] = {m: (judge[m]["mean"] if m in judge else None) for m in LEGACY_KEYS}   # old format

    # ---------------- performance ----------------
    tok_rows = [r for r in ok if (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0) > 0]
    perf = {
        "latency_ms": {n: latency_stats([r.get(n) for r in ok]) for n in ("retrieval_ms", "generation_ms", "total_ms")},
        "tokens_per_query": None, "cost_per_query_usd": None, "cost_per_1k_queries_usd": None,
    }
    if tok_rows:
        tin, tout = mean(r["input_tokens"] for r in tok_rows), mean(r["output_tokens"] for r in tok_rows)
        cost = tin * GENERATOR_PRICE_PER_M["input"] / 1e6 + tout * GENERATOR_PRICE_PER_M["output"] / 1e6
        perf.update(tokens_per_query={"input": tin, "output": tout}, cost_per_query_usd=cost,
                    cost_per_1k_queries_usd=cost * 1000)
    report["performance"] = perf

    # ---------------- by category ----------------
    by_cat = {}
    ev_by_id = {e["id"]: e for e in evaluated}
    ret_by_id = {r["id"]: r for r in retrieval}
    for c in sorted({r["category"] for r in dataset}):
        ids = [r["id"] for r in ok if r["category"] == c]
        d = {"n": len(ids)}
        if c == UNANSWERABLE:
            d["abstention_accuracy"] = mean(float(x["abstained"]) for x in una) if una else None
        else:
            cov = [x["coverage"] for x in ans if x["category"] == c and x["coverage"] is not None]
            d["fact_coverage"] = mean(cov) if cov else None
            cor = [ev_by_id[i]["correctness"] for i in ids if i in ev_by_id and ev_by_id[i].get("correctness") is not None]
            d["correctness"] = mean(cor) if cor else None
            if k:
                hit = [ret_by_id[i]["at"][str(k)]["hit"] for i in ids if i in ret_by_id]
                d["hit"] = mean(hit) if hit else None
                sh = [ret_by_id[i]["at"][str(k)].get("source_recall") for i in ids if i in ret_by_id]
                sh = [x for x in sh if x is not None]
                d["right_doc"] = mean(sh) if sh else None
        by_cat[c] = d
    report["by_category"] = by_cat

    # ---------------- by source document (which file the question is about) ----------------
    src_of = {r["id"]: _src(r.get("source")) for r in dataset}
    by_src = {}
    for sname in sorted(set(src_of.values())):
        ids_all = [r["id"] for r in ok if src_of.get(r["id"]) == sname]
        ids = [r["id"] for r in ok if src_of.get(r["id"]) == sname and r["category"] != UNANSWERABLE]
        cov = [x["coverage"] for x in ans if x["id"] in ids and x["coverage"] is not None]
        cor = [ev_by_id[i]["correctness"] for i in ids if i in ev_by_id and ev_by_id[i].get("correctness") is not None]
        hit = [ret_by_id[i]["at"][str(k)]["hit"] for i in ids if i in ret_by_id and k]
        una_s = [float(x["abstained"]) for x in una if x["id"] in ids_all]
        by_src[sname] = {"n": len(ids_all), "hit": mean(hit) if hit else None,
                         "fact_coverage": mean(cov) if cov else None, "correctness": mean(cor) if cor else None,
                         "abstention_accuracy": mean(una_s) if una_s else None}
    report["by_source"] = by_src
    return report


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(s, nd=3):
    return "n/a" if not s else f"{s['mean']:.{nd}f}  [{s['ci_low']:.{nd}f} - {s['ci_high']:.{nd}f}]"


def print_report(rep: dict) -> None:
    d = rep["dataset"]
    print(f"\n==================== {rep['experiment']} ====================")
    print(f"k = {rep['retriever_k']} | chunking: {rep['chunking']} | judge: {rep['judge_model']} "
          f"| judge context: {rep['context_source']}")
    print(f"corpus: {', '.join(rep['corpus']['documents'])} | vector DB: {rep['corpus']['vector_db']}")
    print(f"questions: {d['questions_in_dataset']} ({d['by_category']}) | answers generated: {d['answers_generated']} "
          f"| errors: {d['generation_errors']} | judged: {d['judged']}")

    print("\n--- Judge metrics, answerable questions: mean [95% bootstrap CI] ---")
    if rep["judge"]:
        for m, s in rep["judge"].items():
            print(f"{m.replace('_', ' ').title():<20}{_ci(s)}   pass rate {s['pass_rate']:.0%}  (n={s['n']})")
    else:
        print("no judge scores yet (run eval_all_metrics)")

    print("\n--- Retrieval (judge-free) ---")
    r = rep["retrieval"]
    if r and r["at_k"]:
        for m, label in (("hit", "Hit rate"), ("fact_recall", "Recall"), ("precision", "Precision"), ("mrr", "MRR"),
                         ("source_hit", "Right-doc hit"), ("source_recall", "All-docs reached"),
                         ("source_precision", "Right-doc precision")):
            if r["at_k"].get(m):
                print(f"{label + '@' + str(rep['retriever_k']):<22}{_ci(r['at_k'][m])}")
        print("curve (k: hit / recall / precision / mrr): " + " | ".join(
            f"{kk}: {_f(v['hit'], 2)}/{_f(v['fact_recall'], 2)}/{_f(v['precision'], 2)}/{_f(v['mrr'], 2)}"
            for kk, v in r["curve"].items()))
        lat = r["latency_ms"]
        print(f"retrieval latency   p50 {_f(lat['p50'], 1)} ms | p95 {_f(lat['p95'], 1)} ms")
        if r["misses"]:
            print(f"questions never retrieved within top {max(map(int, r['curve']))}: {', '.join(r['misses'])}")
    else:
        print("no retrieval results yet (run retrieval_eval)")

    g = rep["generation"]
    print("\n--- Generation (judge-free) ---")
    print(f"{'Key-fact coverage':<26}{_ci(g['fact_coverage'])}")
    print(f"{'Abstention accuracy':<26}{_ci(g['abstention_accuracy'])}   (unanswerable: said 'not in document')")
    print(f"{'Hallucination rate':<26}{_ci(g['hallucination_rate'])}   (unanswerable: answered anyway)")
    print(f"{'False refusal rate':<26}{_ci(g['false_refusal_rate'])}   (answerable: wrongly refused)")
    print(f"{'Context grounding':<26}{_ci(g['context_grounding'])}   (LLM context found in retrieved chunks)")
    print(f"{'Paraphrase consistency':<26}{_f(g['paraphrase_consistency'])}   (answers across rephrasings)")
    print(f"{'Paraphrase retr. consist.':<26}{_f(g['paraphrase_retrieval_consistency'])}")

    pf = rep["performance"]
    print("\n--- Performance ---")
    for n in ("retrieval_ms", "generation_ms", "total_ms"):
        s = pf["latency_ms"][n]
        if s:
            print(f"{n.replace('_ms', '').title() + ' latency':<26}p50 {s['p50']:.0f} ms | p95 {s['p95']:.0f} ms | mean {s['mean']:.0f} ms")
    if pf["tokens_per_query"]:
        t = pf["tokens_per_query"]
        print(f"{'Tokens per query':<26}in {t['input']:.0f} | out {t['output']:.0f}")
        print(f"{'Cost per query':<26}${pf['cost_per_query_usd']:.6f}   (${pf['cost_per_1k_queries_usd']:.3f} per 1,000 queries)")
    else:
        print("token/cost data not available")

    print("\n--- By category ---")
    for c, v in rep["by_category"].items():
        parts = [f"n={v['n']}"] + [f"{k} {_f(x, 2)}" for k, x in v.items() if k != "n"]
        print(f"{c:<16}" + " | ".join(parts))
    print("\n--- By source document ---")
    for sname, v in rep.get("by_source", {}).items():
        parts = [f"n={v['n']}"] + [f"{k} {_f(x, 2)}" for k, x in v.items() if k != "n" and x is not None]
        print(f"{sname:<48}" + " | ".join(parts))
    print()
