"""Qdrant (cloud or in-memory): one collection per version and per exact content (reused when nothing changed)."""
import hashlib
import json
import time
import uuid

from .config import QDRANT_API_KEY, QDRANT_URL

_client = None


def get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        try:
            if QDRANT_URL == ":memory:":
                _client = QdrantClient(":memory:")
            else:
                _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
                _client.get_collections()
        except Exception as e:
            raise SystemExit(f"Cannot reach Qdrant at {QDRANT_URL} ({type(e).__name__}: {e}).\n"
                             "Check QDRANT_URL / QDRANT_API_KEY in backend/.env (or set QDRANT_URL=:memory: to test).")
    return _client


def fingerprint(version: str, spec: dict, chunks) -> str:
    blob = json.dumps({"v": version, "chunking": spec["chunking"], "embedding": spec["embedding"],
                       "texts": [c["text"] for c in chunks]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def build_index(version: str, spec: dict, chunks, embedder):
    """Returns the collection name. Creates and fills it only if this exact content is not indexed yet."""
    from qdrant_client.http.models import Distance, PointStruct, VectorParams
    client = get_client()
    name = f"asterion_{version}_{fingerprint(version, spec, chunks)}"
    if client.collection_exists(name) and client.count(name, exact=True).count == len(chunks):
        print(f"{version}: reusing Qdrant collection {name} ({len(chunks)} points)")
        return name

    if client.collection_exists(name):
        client.delete_collection(name)
    print(f"{version}: embedding {len(chunks)} chunks with {spec['embedding']['model']} ({spec['embedding']['dim']} dims) ...")
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    client.create_collection(name, vectors_config=VectorParams(size=spec["embedding"]["dim"], distance=Distance.COSINE))
    points = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{c['id']}")), vector=v,
                          payload={"i": i, **c}) for i, (c, v) in enumerate(zip(chunks, vectors))]
    batch = 8                                    # small batches: big uploads time out on slow connections
    for s in range(0, len(points), batch):
        for attempt in range(5):
            try:
                client.upsert(name, points=points[s:s + batch], wait=True)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"  upsert retry {attempt + 1}/5 ({type(e).__name__}) ...")
                time.sleep(3 * (attempt + 1))
    print(f"{version}: indexed {len(points)} points in {name}")
    return name


def dense_search(collection: str, vector, limit: int):
    """[(chunk_index, score)] best first."""
    res = get_client().query_points(collection_name=collection, query=vector, limit=limit, with_payload=["i"])
    return [(int(p.payload["i"]), float(p.score)) for p in res.points]


