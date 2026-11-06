"""Judge-free generation metrics: key-fact coverage, abstention, grounding."""
import re

from .text_utils import normalize, tokens

# Phrases that signal "I could not find this in the context" (checked on normalized text).
ABSTAIN_PHRASES = [
    "does not provide", "does not contain", "does not include", "does not mention", "does not specify",
    "does not list", "does not state", "do not provide", "do not contain", "doesn't provide", "doesn't contain",
    "doesn't include", "doesn't mention", "doesn't specify", "not available", "not provided", "not mentioned",
    "not specified", "not stated", "not listed", "no information", "no details", "no mention", "cannot find",
    "can't find", "couldn't find", "unable to find", "not in the document", "not in the retrieved",
    "insufficient information", "not enough information",
    "does not indicate", "doesn't indicate", "no indication", "not indicated", "does not say", "doesn't say",
]


def is_abstention(answer) -> bool:
    a = normalize(answer)
    return any(p in a for p in ABSTAIN_PHRASES)


def fact_coverage(answer, gold_facts):
    """Share of gold facts present in the answer. Each fact is a list of acceptable alternatives."""
    if not gold_facts:
        return None
    a = normalize(answer)
    hit = sum(any(normalize(alt) in a for alt in alts) for alts in gold_facts)
    return hit / len(gold_facts)


def ngram_grounding(context, chunks, n=3):
    """Share of the LLM-written `context` word n-grams that really occur in the retrieved chunks.
    Low values mean the model invented or heavily rewrote its 'supporting context'."""
    context = re.sub(r"\[source:[^\]]*\]", " ", context or "", flags=re.I)   # drop the "[Source: file, page n]" labels
    ctx = tokens(context)
    if not ctx:
        return None
    pool = tokens(" ".join(chunks))
    if len(ctx) < n:
        return 1.0 if " ".join(ctx) in " ".join(pool) else 0.0
    grams = {tuple(pool[i:i + n]) for i in range(len(pool) - n + 1)}
    ctx_grams = [tuple(ctx[i:i + n]) for i in range(len(ctx) - n + 1)]
    return sum(g in grams for g in ctx_grams) / len(ctx_grams)
