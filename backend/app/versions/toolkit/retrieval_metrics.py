"""Retrieval metrics against gold `evidence` snippets (verbatim pieces of the PDF).

A retrieved chunk counts as RELEVANT if it contains at least one evidence snippet.
Judge-free, deterministic and free of API cost.
"""
from .text_utils import normalize


def metrics_at_k(chunks, evidence, k):
    top = [normalize(c) for c in chunks[:k]]
    ev = [normalize(e) for e in evidence]

    relevant = [any(e in c for e in ev) for c in top]
    found = [any(e in c for c in top) for e in ev]

    first_rank = next((i for i, r in enumerate(relevant, 1) if r), None)

    return {
        "hit": float(any(relevant)),                                  # Hit rate @k
        "fact_recall": sum(found) / len(ev) if ev else None,          # Recall@k (share of evidence snippets found)
        "precision": sum(relevant) / len(top) if top else 0.0,        # Precision@k
        "mrr": (1.0 / first_rank) if first_rank else 0.0,             # reciprocal rank (cut at k)
    }


def retrieval_record(chunks, evidence, depths):
    ev = [normalize(e) for e in evidence]
    norm = [normalize(c) for c in chunks]
    first_rank = next((i for i, c in enumerate(norm, 1) if any(e in c for e in ev)), None)
    return {
        "first_hit_rank": first_rank,
        "at": {str(k): metrics_at_k(chunks, evidence, k) for k in depths},
    }
