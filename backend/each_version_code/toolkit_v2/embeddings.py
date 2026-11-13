"""Gemini embeddings through the API (no model is downloaded) with a disk cache.

Every text is embedded once per (model, dimension): re-running an experiment costs no API calls and no quota.
"""
import hashlib
import json
import math
import re
import sqlite3
import threading
import time

from .config import CACHE_DIR, EMBED_BATCH, EMBED_PAUSE, GEMINI_API_KEY

DOC_PREFIX = "title: none | text: "          # used only if the model rejects task_type (see _call)
QUERY_PREFIX = "task: search result | query: "


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]                 # Google asks to L2-normalize truncated (non-3072) vectors


class GeminiEmbedder:
    def __init__(self, model: str, dim: int):
        self.model, self.dim = model, dim
        self.mode = "task_type"               # falls back to "prefix" if the API rejects task_type for this model
        self._client = None
        self._lock = threading.Lock()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(CACHE_DIR / "embeddings.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS e (k TEXT PRIMARY KEY, v TEXT)")

    # ---- cache
    def _key(self, kind, text):
        return hashlib.sha1(f"{self.model}|{self.dim}|{kind}|{text}".encode("utf-8")).hexdigest()

    def _get(self, key):
        with self._lock:
            row = self._db.execute("SELECT v FROM e WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, items):
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO e VALUES (?,?)", [(k, json.dumps(v)) for k, v in items])
            self._db.commit()

    # ---- API
    @property
    def client(self):
        if self._client is None:
            if not GEMINI_API_KEY:
                raise SystemExit("GEMINI_API_KEY is missing. Add it to backend/.env (get one at aistudio.google.com/apikey).")
            from google import genai
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        return self._client

    def _call(self, texts, kind):
        from google.genai import types
        task = "RETRIEVAL_DOCUMENT" if kind == "doc" else "RETRIEVAL_QUERY"
        for attempt in range(1, 9):
            try:
                if self.mode == "task_type":
                    cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
                    contents = texts
                else:
                    cfg = types.EmbedContentConfig(output_dimensionality=self.dim)
                    contents = [(DOC_PREFIX if kind == "doc" else QUERY_PREFIX) + t for t in texts]
                res = self.client.models.embed_content(model=self.model, contents=contents, config=cfg)
                vecs = [list(e.values) for e in res.embeddings]
                if len(vecs) != len(texts):       # some models merge a list into ONE embedding: go one by one
                    vecs = []
                    for t in contents:
                        r = self.client.models.embed_content(model=self.model, contents=t, config=cfg)
                        vecs.append(list(r.embeddings[0].values))
                return [_normalize(v) for v in vecs]
            except Exception as e:
                msg = str(e)
                if self.mode == "task_type" and re.search(r"task[_ ]type", msg, re.I):
                    print(f"  [{self.model}] task_type not accepted, switching to prompt prefixes")
                    self.mode = "prefix"
                    continue
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg:
                    m = re.search(r"retry in ([\d.]+)s", msg, re.I)
                    wait = float(m.group(1)) + 1 if m else min(60, 5 * attempt)
                    print(f"  [{self.model}] rate limit/unavailable, retry {attempt}/8 in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Gemini embedding failed after 8 retries")

    def _embed(self, texts, kind):
        kind_key = f"{kind}:{self.mode}"
        out = [self._get(self._key(kind_key, t)) for t in texts]
        todo = [i for i, v in enumerate(out) if v is None]
        for s in range(0, len(todo), EMBED_BATCH):
            idx = todo[s:s + EMBED_BATCH]
            vecs = self._call([texts[i] for i in idx], kind)
            kind_key = f"{kind}:{self.mode}"      # mode may have switched during the call
            self._put([(self._key(kind_key, texts[i]), v) for i, v in zip(idx, vecs)])
            for i, v in zip(idx, vecs):
                out[i] = v
            if len(todo) > EMBED_BATCH:
                print(f"  embedded {min(s + EMBED_BATCH, len(todo))}/{len(todo)}")
                time.sleep(EMBED_PAUSE)
        return out

    def embed_documents(self, texts):
        return self._embed(list(texts), "doc")

    def embed_query(self, text):
        return self._embed([text], "query")[0]


_cache = {}


def get_embedder(model: str, dim: int) -> GeminiEmbedder:
    if (model, dim) not in _cache:
        _cache[(model, dim)] = GeminiEmbedder(model, dim)
    return _cache[(model, dim)]
