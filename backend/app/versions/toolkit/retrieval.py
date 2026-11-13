
"""Retrieval strategies and the pipeline factory.

  vector         : dense search, top_k                                    (v1)
  vector_rerank  : dense search fetch_k, rerank, keep top_k               (v2)
  hybrid_rerank  : dense + BM25 fused with RRF, child -> parent, rerank   (v3)

Debug mode:
  - Inspect dense retrieval results.
  - Inspect child and parent context sizes.
  - Inspect reranking results.
  - Inspect exact final context sent to the LLM.
"""

import re
import time

from .config import VERSIONS
from .corpus import describe, make_chunks
from .embeddings import get_embedder
from .rerank import get_reranker
from .vectorstore import build_index, dense_search

TOKEN = re.compile(r"[a-z0-9₹@._]+", re.IGNORECASE)


def _tok(text):
    return TOKEN.findall(text.lower())


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion of several ranked lists of ids."""
    score = {}

    for ranking in rankings:
        for rank, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)

    return [
        i for i, _ in sorted(
            score.items(),
            key=lambda x: -x[1]
        )
    ]


class Retriever:

    def __init__(self, version, spec, chunks, collection, embedder):
        self.version = version
        self.spec = spec
        self.chunks = chunks
        self.collection = collection
        self.embedder = embedder

        self.mode = spec["retrieval"]["mode"]

        self.reranker = (
            get_reranker()
            if "rerank" in self.mode
            else None
        )

        self.bm25 = None

        if self.mode == "hybrid_rerank":
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(
                [_tok(c["text"]) for c in chunks]
            )

    def _context(self, chunk):
        """Context passed to the LLM."""

        return {
            "text": chunk.get("parent_text") or chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"]
        }

    # --------------------------------------------------
    # DEBUG HELPER
    # --------------------------------------------------

    def _debug_chunk(
        self,
        label,
        chunk,
        rank=None,
        score=None,
        index=None
    ):
        """Print chunk information for retrieval analysis."""

        child_text = chunk.get("text", "")
        parent_text = chunk.get("parent_text") or child_text

        print("\n" + "-" * 90)

        print(f"{label}")

        if rank is not None:
            print(f"Rank          : {rank}")

        if index is not None:
            print(f"Chunk index   : {index}")

        print(f"Chunk ID      : {chunk.get('id', 'unknown')}")
        print(f"Parent ID     : {chunk.get('parent_id', 'N/A')}")
        print(f"Page          : {chunk.get('page', 'unknown')}")
        print(f"Source        : {chunk.get('source', 'unknown')}")

        if score is not None:
            print(f"Score         : {score}")

        print(f"Child chars   : {len(child_text)}")
        print(f"Context chars : {len(parent_text)}")

        print(f"Child tokens  : {len(_tok(child_text))}")
        print(f"Context tokens: {len(_tok(parent_text))}")

        print("\n--- ACTUAL CONTEXT SENT TO LLM ---")
        print(parent_text)

        print("-" * 90)

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

    def retrieve(self, question, debug=False):

        r = self.spec["retrieval"]
        timings = {}

        if debug:
            print("\n" + "=" * 100)
            print(f"VERSION       : {self.version}")
            print(f"QUESTION      : {question}")
            print(f"MODE          : {self.mode}")
            print(f"TOP K         : {r['top_k']}")
            print(f"COLLECTION    : {self.collection}")
            print("=" * 100)

        # ----------------------------------------------
        # 1. Query embedding
        # ----------------------------------------------

        t0 = time.perf_counter()

        qvec = self.embedder.embed_query(question)

        timings["embed_ms"] = (
            time.perf_counter() - t0
        ) * 1000

        # ----------------------------------------------
        # 2. Dense retrieval
        # ----------------------------------------------

        t0 = time.perf_counter()

        fetch = r.get("fetch_k", r["top_k"])

        dense_hits = dense_search(
            self.collection,
            qvec,
            fetch
        )

        dense = [i for i, _ in dense_hits]

        if debug:
            print("\n\n### STEP 1: DENSE RETRIEVAL")
            print(f"Retrieved candidates: {len(dense_hits)}")

            for rank, (idx, score) in enumerate(
                dense_hits,
                start=1
            ):
                self._debug_chunk(
                    label="DENSE RESULT",
                    chunk=self.chunks[idx],
                    rank=rank,
                    score=round(score, 5),
                    index=idx
                )

        # ----------------------------------------------
        # 3. V1: Vector retrieval only
        # ----------------------------------------------

        if self.mode == "vector":

            ranked = dense[:r["top_k"]]

            contexts = [
                self._context(self.chunks[i])
                for i in ranked
            ]

            timings["search_ms"] = (
                time.perf_counter() - t0
            ) * 1000

            if debug:
                print("\n\n### STEP 2: FINAL CONTEXT")

                for rank, idx in enumerate(
                    ranked,
                    start=1
                ):
                    self._debug_chunk(
                        label="FINAL CONTEXT",
                        chunk=self.chunks[idx],
                        rank=rank,
                        index=idx
                    )

                self._debug_summary(contexts)

            return contexts, timings

        # ----------------------------------------------
        # 4. V3: Hybrid retrieval
        # ----------------------------------------------

        if self.mode == "hybrid_rerank":

            scores = self.bm25.get_scores(
                _tok(question)
            )

            sparse = sorted(
                range(len(scores)),
                key=lambda i: -scores[i]
            )[:fetch]

            candidates = rrf([
                dense,
                sparse
            ])

            if debug:
                print("\n\n### STEP 2: HYBRID RETRIEVAL")
                print(f"Dense candidates : {len(dense)}")
                print(f"Sparse candidates: {len(sparse)}")
                print(f"Fused candidates : {len(candidates)}")

        # ----------------------------------------------
        # 5. V2: Dense retrieval candidates
        # ----------------------------------------------

        else:
            candidates = dense

        # ----------------------------------------------
        # 6. Build reranking pool
        # ----------------------------------------------

        pool = []
        seen = set()

        for i in candidates:

            chunk = self.chunks[i]

            key = chunk.get("parent_id") or i

            # Avoid duplicate parents
            if key in seen:
                continue

            seen.add(key)

            pool.append(
                self._context(chunk)
            )

            if len(pool) >= r.get(
                "rerank_pool",
                fetch
            ):
                break

        if debug:
            print("\n\n### STEP 3: RERANKING POOL")

            for rank, context in enumerate(
                pool,
                start=1
            ):
                print(f"\nPool rank: {rank}")
                print(f"Source: {context['source']}")
                print(f"Page: {context['page']}")
                print(f"Characters: {len(context['text'])}")
                print(f"Tokens: {len(_tok(context['text']))}")
                print(context["text"])

        timings["search_ms"] = (
            time.perf_counter() - t0
        ) * 1000

        # ----------------------------------------------
        # 7. Reranking
        # ----------------------------------------------

        t0 = time.perf_counter()

        order = self.reranker.rerank(
            question,
            [c["text"] for c in pool]
        )

        timings["rerank_ms"] = (
            time.perf_counter() - t0
        ) * 1000

        # ----------------------------------------------
        # 8. Reranking inspection
        # ----------------------------------------------

        if debug:
            print("\n\n### STEP 4: RERANKING RESULTS")

            for rank, (idx, score) in enumerate(
                order,
                start=1
            ):
                context = pool[idx]

                print("\n" + "-" * 80)
                print(f"Reranked position: {rank}")
                print(f"Original pool index: {idx}")
                print(f"Rerank score: {score}")
                print(f"Characters: {len(context['text'])}")
                print(f"Tokens: {len(_tok(context['text']))}")
                print(context["text"])

        # ----------------------------------------------
        # 9. Final contexts
        # ----------------------------------------------

        final_order = order[:r["top_k"]]

        final_contexts = [
            pool[i]
            for i, _ in final_order
        ]

        if debug:
            print("\n\n### STEP 5: FINAL CONTEXT SENT TO LLM")

            for rank, (idx, score) in enumerate(
                final_order,
                start=1
            ):
                context = pool[idx]

                print("\n" + "-" * 80)
                print(f"Final rank: {rank}")
                print(f"Rerank score: {score}")
                print(f"Source: {context['source']}")
                print(f"Page: {context['page']}")
                print(f"Characters: {len(context['text'])}")
                print(f"Tokens: {len(_tok(context['text']))}")
                print("\nCONTENT:")
                print(context["text"])

            self._debug_summary(final_contexts)

        return final_contexts, timings

    # --------------------------------------------------
    # FINAL CONTEXT SUMMARY
    # --------------------------------------------------

    def _debug_summary(self, contexts):

        total_chars = sum(
            len(c["text"])
            for c in contexts
        )

        total_tokens = sum(
            len(_tok(c["text"]))
            for c in contexts
        )

        print("\n\n### CONTEXT SUMMARY")

        print(f"Number of contexts : {len(contexts)}")
        print(f"Total characters  : {total_chars}")
        print(f"Approx. tokens    : {total_tokens}")

        if contexts:
            print(
                f"Average chars     : "
                f"{total_chars / len(contexts):.0f}"
            )

            print(
                f"Average tokens    : "
                f"{total_tokens / len(contexts):.0f}"
            )

        print("=" * 100)


# ------------------------------------------------------
# PIPELINE FACTORY
# ------------------------------------------------------

def build_pipeline(version: str) -> Retriever:

    spec = VERSIONS[version]

    embedder = get_embedder(
        spec["embedding"]["model"],
        spec["embedding"]["dim"]
    )

    chunks = make_chunks(
        spec,
        embedder
    )

    print(
        f"{version}: {describe(chunks)} "
        f"| {spec['chunking']['strategy']} "
        f"| {spec['embedding']['model']} "
        f"| {spec['retrieval']['mode']} "
        f"top_k={spec['retrieval']['top_k']}"
    )

    collection = build_index(
        version,
        spec,
        chunks,
        embedder
    )

    return Retriever(
        version,
        spec,
        chunks,
        collection,
        embedder
    )
