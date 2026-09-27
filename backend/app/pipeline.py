"""
Ties LangChain-based chunking + embedding + Qdrant together into one
configurable retrieval pipeline. `PipelineConfig` is deliberately the
object you'll sweep over in Phase 3 (the Experiment Engine) — nothing
about this file needs to change later, only the config values passed in.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()  # reads .env (GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, etc.) into os.environ

from chunking import chunk_documents
from embedder import get_embedder
from vector_store import QdrantStore
from generator import get_generator


@dataclass
class PipelineConfig:
    chunk_size: int = 400       # characters (LangChain splitter convention)
    chunk_overlap: int = 80
    embedder_name: str = "bge-small-en-v1.5"   # one of: bge-m3, qwen3-embedding-0.6b, bge-small-en-v1.5
    top_k: int = 5
    collection_name: str = "rag_chunks"
    generator_name: str = "groq"               # only "groq" is supported

    def as_dict(self):
        return asdict(self)


class RetrievalPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.embedding = get_embedder(config.embedder_name)
        self.store = QdrantStore(embedding=self.embedding, collection_name=config.collection_name)
        self.generator = get_generator(config.generator_name)
        self.chunks: List[Document] = []

    def index(self, documents: Dict[str, str]) -> None:
        """Chunk and index the full document set."""
        self.chunks = chunk_documents(
            documents, chunk_size=self.config.chunk_size, overlap=self.config.chunk_overlap
        )
        self.store.build(self.chunks)

    def retrieve(self, query: str, top_k: int = None) -> List[dict]:
        """Retrieve the top_k most relevant chunks for a query."""
        k = top_k or self.config.top_k
        return self.store.search(query, top_k=k)

    def chunk_ids_for_docs(self, doc_ids: List[str]) -> set:
        """Helper: resolve a list of doc_ids into the set of chunk_ids belonging to them."""
        doc_id_set = set(doc_ids)
        return {c.metadata["chunk_id"] for c in self.chunks if c.metadata["doc_id"] in doc_id_set}

    def answer(self, question: str, top_k: int = None) -> dict:
        """
        Phase 2: full retrieval + generation for one question.
        Returns {"question", "retrieved_hits", "context", "answer"}.
        """
        hits = self.retrieve(question, top_k=top_k)
        context = [h["text"] for h in hits]
        generated = self.generator.generate(question, context)
        return {
            "question": question,
            "retrieved_hits": hits,
            "context": context,
            "answer": generated,
        }
