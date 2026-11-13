"""Compare retrieval setups WITHOUT Groq: no LLM answers and no judge. Uses only Gemini embeddings (cached) and FlashRank.

Run from backend/:   python -m app.evaluation.retrieval_check
Everything is indexed in memory (nothing is written to Qdrant Cloud). Edit CANDIDATES to try your own ideas.

Columns
  any-hit   share of answerable questions where at least one gold evidence snippet is in the retrieved chunks
  all-hit   share of multi-part / multi-hop questions where EVERY gold evidence snippet is in the retrieved chunks
  ctx chars average characters sent to the LLM (smaller = less noise, so higher context relevancy)
"""
import os
from collections import defaultdict

os.environ["QDRANT_URL"] = ":memory:"          # must be set before the toolkit is imported

from app.versions.toolkit.config import EMBEDDING_1, VERSIONS, paths          # noqa: E402
from app.versions.toolkit.retrieval import build_pipeline                      # noqa: E402
from app.versions.toolkit.text_utils import normalize, read_jsonl              # noqa: E402

STATIC = lambda size, overlap: {"strategy": "static", "chunk_chars": size, "overlap_chars": overlap}
E1 = {"model": EMBEDDING_1, "dim": 768}

CANDIDATES = {
    "A  v1 (2000, vector, top 3)":        {"chunking": STATIC(2000, 200), "embedding": E1, "retrieval": {"mode": "vector", "top_k": 3}},
    "B  v1 + rerank (fetch 10, top 3)":   {"chunking": STATIC(2000, 200), "embedding": E1, "retrieval": {"mode": "vector_rerank", "fetch_k": 10, "top_k": 3}},
    "C  1000 chars, vector, top 4":       {"chunking": STATIC(1000, 150), "embedding": E1, "retrieval": {"mode": "vector", "top_k": 4}},
    "D  1000 chars + rerank (15 -> 4)":   {"chunking": STATIC(1000, 150), "embedding": E1, "retrieval": {"mode": "vector_rerank", "fetch_k": 15, "top_k": 4}},
    "E  700 chars + rerank (20 -> 5)":    {"chunking": STATIC(700, 100),  "embedding": E1, "retrieval": {"mode": "vector_rerank", "fetch_k": 20, "top_k": 5}},
}


def main():
    rows = [r for r in read_jsonl(paths("v1")["dataset"]) if r["category"] != "unanswerable"]
    print(f"{len(rows)} answerable questions\n")
    table = []
    for i, (label, spec) in enumerate(CANDIDATES.items()):
        name = f"try{i}"
        VERSIONS[name] = {"n": 99, "name": name, "label": label, **spec}
        print(f"--- {label}")
        retriever = build_pipeline(name)
        any_hit, all_hit, chars = [], [], []
        by_cat = defaultdict(list)
        for r in rows:
            contexts, _ = retriever.retrieve(r["question"])
            texts = [normalize(c["text"]) for c in contexts]
            found = [any(normalize(e["text"]) in t for t in texts) for e in r["evidence"]]
            hit = int(any(found))
            any_hit.append(hit)
            by_cat[r["category"]].append(hit)
            if len(found) > 1:
                all_hit.append(int(all(found)))
            chars.append(sum(len(c["text"]) for c in contexts))
        table.append((label, sum(any_hit) / len(any_hit), sum(all_hit) / max(len(all_hit), 1),
                      sum(chars) / len(chars), {k: sum(v) / len(v) for k, v in by_cat.items()}))

    print("\n" + "=" * 96)
    print(f"{'setup':38} {'any-hit':>8} {'all-hit':>8} {'ctx chars':>10}   by type (any-hit)")
    for label, a, al, ch, cats in table:
        detail = "  ".join(f"{k}:{v:.2f}" for k, v in cats.items())
        print(f"{label:38} {a:8.3f} {al:8.3f} {ch:10.0f}   {detail}")


if __name__ == "__main__":
    main()