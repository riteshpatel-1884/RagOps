"""v1 agent: exposes ask(question) -> dict (used by main.py and the evaluation runner)."""
from app.versions.toolkit.rag_core import make_ask

from .indexing import pipeline

ask = make_ask(pipeline)
