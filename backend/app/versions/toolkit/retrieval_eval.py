"""Retrieval-only evaluation: no LLM, no API key, no cost.

Runs the version's retriever on every answerable question and scores it against the gold
`evidence` snippets: Hit rate, Recall, Precision and MRR at k = 1, 3, 5 (and the version's own k),
plus retrieval latency.
"""
import importlib
import time
from statistics import median

from .config import RETRIEVAL_DEPTHS, paths, rag_module, UNANSWERABLE
from .retrieval_metrics import retrieval_record
from .text_utils import read_jsonl, write_jsonl


def main(version: str):
    p = paths(version)
    cases = [c for c in read_jsonl(p["dataset"]) if c["category"] != UNANSWERABLE and c.get("evidence")]
    if not cases:
        raise SystemExit(f"Dataset missing or empty: {p['dataset']}")

    idx = importlib.import_module(rag_module(version, "indexing"))
    store, k_own = idx.vector_store, idx.RETRIEVER_K
    depths = sorted(set(RETRIEVAL_DEPTHS) | {k_own})
    k_max = max(depths)

    store.similarity_search("warm-up query", k=k_max)   # first call loads the model; keep it out of latency

    rows = []
    for c in cases:
        times, docs = [], None
        for _ in range(3):                               # median of 3 runs = stable latency
            t0 = time.perf_counter()
            docs = store.similarity_search(c["question"], k=k_max)
            times.append((time.perf_counter() - t0) * 1000)
        chunks = [d.page_content for d in docs]
        rows.append({
            "id": c["id"], "category": c["category"], "paraphrase_group": c.get("paraphrase_group"),
            "question": c["question"], "retriever_k": k_own, "retrieval_ms": median(times),
            **retrieval_record(chunks, c["evidence"], depths),
            "retrieved_chunks": chunks,
        })

    write_jsonl(p["retrieval"], rows)

    print(f"\n{version}: retrieval on {len(rows)} answerable questions (own k = {k_own})")
    print(f"{'k':<4}{'Hit rate':<10}{'Recall':<10}{'Precision':<11}{'MRR':<8}")
    for k in depths:
        def avg(m):
            v = [r["at"][str(k)][m] for r in rows if r["at"][str(k)][m] is not None]
            return sum(v) / len(v)
        mark = "  <- this version's k" if k == k_own else ""
        print(f"{k:<4}{avg('hit'):<10.3f}{avg('fact_recall'):<10.3f}{avg('precision'):<11.3f}{avg('mrr'):<8.3f}{mark}")
    print(f"\nSaved: {p['retrieval']}")
