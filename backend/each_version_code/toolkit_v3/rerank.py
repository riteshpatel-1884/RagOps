"""Rerankers. FlashRank = tiny CPU cross-encoder (a ~4 MB file is downloaded once). LLM = Groq ranks the passages."""
import json
import re

from .config import CACHE_DIR, FLASHRANK_MODEL, LLM_MODEL, RERANKER


class FlashRankReranker:
    name = "flashrank"

    def __init__(self):
        from flashrank import Ranker
        self.ranker = Ranker(model_name=FLASHRANK_MODEL, cache_dir=str(CACHE_DIR / "flashrank"))

    def rerank(self, query, texts):
        from flashrank import RerankRequest
        res = self.ranker.rerank(RerankRequest(query=query, passages=[{"id": i, "text": t} for i, t in enumerate(texts)]))
        return [(int(r["id"]), float(r["score"])) for r in res]          # already sorted best first


class LLMReranker:
    name = "llm"

    def __init__(self):
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(model=LLM_MODEL, temperature=0)

    def rerank(self, query, texts):
        listing = "\n\n".join(f"[{i}] {t[:1200]}" for i, t in enumerate(texts))
        prompt = (f"Question: {query}\n\nPassages:\n{listing}\n\n"
                  'Rank the passages by how useful they are for answering the question. '
                  'Reply with JSON only: {"order": [ids, best first]}')
        raw = self.llm.invoke(prompt).content
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            order = [int(i) for i in json.loads(m.group(0))["order"]] if m else []
        except Exception:
            order = []
        seen, ranked = set(), []
        for i in order + list(range(len(texts))):
            if 0 <= i < len(texts) and i not in seen:
                seen.add(i)
                ranked.append((i, float(len(texts) - len(ranked))))
        return ranked


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = LLMReranker() if RERANKER == "llm" else FlashRankReranker()
    return _reranker
