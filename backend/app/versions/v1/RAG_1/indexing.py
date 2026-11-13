"""v1: builds (or reuses) this version's Qdrant index. What it does is defined in toolkit/config.py (VERSIONS["v1"])."""
from app.versions.toolkit.retrieval import build_pipeline

pipeline = build_pipeline("v1")
