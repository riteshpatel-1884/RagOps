"""Judge-free metrics: hit rate @k, abstention (false refusal / hallucination)."""
from .config import HIT_K
from .text_utils import normalize

ABSTAIN_PHRASES = [
    "could not find", "couldn't find", "cannot find", "can't find", "unable to find", "not found in",
    "does not provide", "does not contain", "does not include", "does not mention", "does not specify",
    "does not list", "does not state", "does not say", "does not indicate", "do not provide", "do not contain",
    "doesn't provide", "doesn't contain", "doesn't include", "doesn't mention", "doesn't specify",
    "not available", "not provided", "not mentioned", "not specified", "not stated", "not listed",
    "no information", "no details", "no mention", "no indication", "not indicated", "not in the handbook",
    "not in the context", "insufficient information", "not enough information",
]


def is_abstention(answer) -> bool:
    a = normalize(answer)
    return any(p in a for p in ABSTAIN_PHRASES)


def hit_at_k(chunks, evidence, k=HIT_K):
    """1 if ANY gold evidence snippet appears inside one of the first k retrieved chunks, else 0."""
    texts = [normalize(c) for c in (chunks or [])[:k]]
    return int(any(normalize(e["text"]) in t for e in evidence for t in texts))
