"""Retrieval strategies and the pipeline factory.

  vector         : dense search, top_k                                    (v1)
  vector_rerank  : dense search fetch_k, rerank, keep top_k               (v2)
  hybrid_rerank  : dense + BM25 fused with RRF, child -> parent, rerank   (v3)
"""
import re
import time

from .config import VERSIONS
from .corpus import describe, make_chunks
from .embeddings import get_embedder
from .rerank import get_reranker
from .vectorstore import build_index, dense_search

TOKEN = re.compile(r"[a-z0-9₹@._]+")


def _tok(text):
    return TOKEN.findall(text.lower())


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion of several ranked lists of ids."""
    score = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])]


class Retriever:
    def __init__(self, version, spec, chunks, collection, embedder):
        self.version, self.spec, self.chunks, self.collection, self.embedder = version, spec, chunks, collection, embedder
        self.mode = spec["retrieval"]["mode"]
        self.reranker = get_reranker() if "rerank" in self.mode else None
        self.bm25 = None
        if self.mode == "hybrid_rerank":
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])

    def _context(self, chunk):
        """What the LLM reads for this chunk: the parent text for parent-child chunks, else the chunk itself."""
        return {"text": chunk.get("parent_text") or chunk["text"], "page": chunk["page"], "source": chunk["source"]}

    def retrieve(self, question):
        r, t = self.spec["retrieval"], {}
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(question)
        t["embed_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fetch = r.get("fetch_k", r["top_k"])
        dense = [i for i, _ in dense_search(self.collection, qvec, fetch)]
        if self.mode == "vector":
            ranked = dense[:r["top_k"]]
            contexts = [self._context(self.chunks[i]) for i in ranked]
            t["search_ms"] = (time.perf_counter() - t0) * 1000
            return contexts, t

        if self.mode == "hybrid_rerank":
            scores = self.bm25.get_scores(_tok(question))
            sparse = sorted(range(len(scores)), key=lambda i: -scores[i])[:fetch]
            candidates = rrf([dense, sparse])
        else:
            candidates = dense

        # several children can share a parent: keep each parent once, in ranked order
        pool, seen = [], set()
        for i in candidates:
            key = self.chunks[i].get("parent_id") or i
            if key in seen:
                continue
            seen.add(key)
            pool.append(self._context(self.chunks[i]))
            if len(pool) >= r.get("rerank_pool", fetch):
                break
        t["search_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        order = self.reranker.rerank(question, [c["text"] for c in pool])
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000
        return [pool[i] for i, _ in order[:r["top_k"]]], t


def build_pipeline(version: str) -> Retriever:
    spec = VERSIONS[version]
    embedder = get_embedder(spec["embedding"]["model"], spec["embedding"]["dim"])
    chunks = make_chunks(spec, embedder)
    print(f"{version}: {describe(chunks)} | {spec['chunking']['strategy']} | {spec['embedding']['model']} "
          f"| {spec['retrieval']['mode']} top_k={spec['retrieval']['top_k']}")
    collection = build_index(version, spec, chunks, embedder)
    return Retriever(version, spec, chunks, collection, embedder)
