"""Retrieval-only evaluation: no LLM, no API key, no cost.

Runs the version's retriever on every answerable question and scores it against the gold
`evidence` snippets: Hit rate, Recall, Precision and MRR at k = 1, 3, 5 (and the version's own k),
plus retrieval latency.
"""
import argparse
import importlib
import time
from statistics import median

from .config import RETRIEVAL_DEPTHS, paths, rag_module, UNANSWERABLE
from .migrate import archive_single_doc
from .query import options_tag, prepare_query
from .retrieval_metrics import retrieval_record
from .text_utils import read_jsonl, write_jsonl


def main(version: str):
    ap = argparse.ArgumentParser(description="Retrieval-only evaluation (free, no LLM).")
    ap.add_argument("--strip", metavar="TERM", help="diagnostic: remove this word from every question before searching")
    ap.add_argument("--no-save", action="store_true", help="print the table only; keep the saved retrieval results untouched")
    args = ap.parse_args()
    print(f"[retrieval options] {options_tag() or 'none'}" + (f" | --strip {args.strip!r}" if args.strip else "")
          + (" | --no-save" if args.no_save else ""))

    archive_single_doc(version)
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
            docs = store.similarity_search(prepare_query(c["question"], [args.strip] if args.strip else None), k=k_max)
            times.append((time.perf_counter() - t0) * 1000)
        chunks = [d.page_content for d in docs]
        sources = [(d.metadata or {}).get("source") for d in docs]
        rows.append({
            "id": c["id"], "category": c["category"], "source": c.get("source"),
            "paraphrase_group": c.get("paraphrase_group"),
            "question": c["question"], "retriever_k": k_own, "retrieval_ms": median(times),
            **retrieval_record(chunks, c["evidence"], depths, sources, c.get("source")),
            "retrieved_chunks": chunks, "retrieved_sources": sources,
            "retrieved_pages": [(d.metadata or {}).get("page") for d in docs],
            "retrieval_opts": options_tag() if not args.strip else f"strip={args.strip.lower()}",
        })

    if not args.no_save:
        write_jsonl(p["retrieval"], rows)

    print(f"\n{version}: retrieval on {len(rows)} answerable questions (own k = {k_own})")
    print(f"{'k':<4}{'Hit rate':<10}{'Recall':<10}{'Precision':<11}{'MRR':<8}{'Right doc':<11}{'All docs reached':<17}")
    for k in depths:
        def avg(m):
            v = [r["at"][str(k)][m] for r in rows if r["at"][str(k)].get(m) is not None]
            return sum(v) / len(v) if v else float("nan")
        mark = "  <- this version's k" if k == k_own else ""
        print(f"{k:<4}{avg('hit'):<10.3f}{avg('fact_recall'):<10.3f}{avg('precision'):<11.3f}{avg('mrr'):<8.3f}"
              f"{avg('source_hit'):<11.3f}{avg('source_recall'):<17.3f}{mark}")
    print(f"\nSaved: {p['retrieval']}" if not args.no_save else "\n(not saved: --no-save)")
