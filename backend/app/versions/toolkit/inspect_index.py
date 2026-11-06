"""Look inside Qdrant: which collections exist, how many chunks per document, and a sample of chunks.

    python -m app.versions.toolkit.inspect_index            # every version
    python -m app.versions.toolkit.inspect_index v3 --show  # also print every chunk of v3
    python -m app.versions.toolkit.inspect_index --drop-stale   # delete old ragops_* collections no version uses
"""
import argparse
import importlib
from collections import Counter

from .config import VERSIONS, rag_module
from .vectorstore import fingerprint, get_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", nargs="?", help="v1, v2 or v3 (default: all)")
    ap.add_argument("--show", action="store_true", help="print every chunk")
    ap.add_argument("--drop-stale", action="store_true", help="delete ragops_* collections no version uses any more")
    a = ap.parse_args()

    client = get_client()
    live = set()
    for v in ([a.version] if a.version else list(VERSIONS)):
        idx = importlib.import_module(rag_module(v, "indexing"))
        name = f"{VERSIONS[v]['collection']}_{fingerprint(idx.chunks)}"
        live.add(name)
        pts = client.count(name, exact=True).count
        per_doc = Counter(c["source"] for c in idx.chunks)
        print(f"\n{v}: collection '{name}' | {pts} points | k = {idx.RETRIEVER_K} | {dict(per_doc)}")
        if a.show:
            for i, c in enumerate(idx.chunks):
                print(f"  [{i}] {c['source']} p{c['pages']} ({len(c['text'])} chars): {c['text'][:110]}")

    if a.drop_stale:
        all_live = {f"{cfg['collection']}_{fingerprint(importlib.import_module(rag_module(v, 'indexing')).chunks)}"
                    for v, cfg in VERSIONS.items()}
        for col in client.get_collections().collections:
            if col.name.startswith("ragops_") and col.name not in all_live:
                client.delete_collection(col.name)
                print(f"dropped stale collection {col.name}")


if __name__ == "__main__":
    main()
