"""Qdrant vector store shared by every version.

* Each version has its own collection, named  <collection>_<fingerprint>.
  The fingerprint is a hash of the chunk texts + embedding model, so a changed PDF or chunking
  automatically gets a fresh collection and an unchanged one is simply reused (no re-embedding).
* Point ids are deterministic, so an interrupted indexing run is completed by running again.
* Safe to import from several processes at once (API server + eval scripts): nothing is deleted.
"""
import hashlib
import uuid

from .config import EMBEDDING_DIM, EMBEDDING_MODEL, QDRANT_API_KEY, QDRANT_URL

_embeddings = None
_clients = {}


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_client():
    from qdrant_client import QdrantClient
    if QDRANT_URL not in _clients:
        if QDRANT_URL == ":memory:":
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        try:
            client.get_collections()
        except Exception as e:                                     # connection refused, wrong url, ...
            raise SystemExit(
                f"Cannot reach Qdrant at {QDRANT_URL} ({type(e).__name__}).\n"
                "Start it with:\n"
                "  docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant\n"
                "or set QDRANT_URL=:memory: in backend/.env to use a throw-away in-process Qdrant.") from e
        _clients[QDRANT_URL] = client
    return _clients[QDRANT_URL]


def fingerprint(chunks) -> str:
    h = hashlib.sha1(EMBEDDING_MODEL.encode())
    for c in chunks:
        h.update(f"|{c['source']}|{c['page']}|{c['text']}".encode("utf-8"))
    return h.hexdigest()[:8]


def build_vector_store(collection: str, chunks: list):
    """Create (or reuse) the Qdrant collection for `chunks` and return a LangChain vector store."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client.http.models import Distance, VectorParams

    client, embeddings = get_client(), get_embeddings()
    fp = fingerprint(chunks)
    name = f"{collection}_{fp}"

    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))

    store = QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)
    have = client.count(name, exact=True).count
    if have < len(chunks):                                          # new or half-finished collection
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{i}")) for i in range(len(chunks))]
        store.add_texts(
            texts=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "page": c["page"], "pages": c["pages"], "chunk_id": i}
                       for i, c in enumerate(chunks)],
            ids=ids)
        print(f"[qdrant] indexed {len(chunks)} chunks into '{name}'")
    else:
        print(f"[qdrant] reusing '{name}' ({have} chunks)")
    return store
