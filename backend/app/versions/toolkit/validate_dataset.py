"""Check the evaluation dataset against the PDFs and across versions.

Usage (from backend/):
    python -m app.versions.toolkit.validate_dataset

Catches: evidence/gold facts that are not in the PDFs, evidence spanning two PDF blocks (unfair to v1/v2),
questions whose `source` file is missing, duplicate ids, more than MAX_QUESTIONS questions, and
datasets that differ between version folders (versions must use the SAME questions).
"""
import hashlib
import sys

from .config import MAX_QUESTIONS, PDF_DIR, UNANSWERABLE, VERSIONS, paths
from .corpus import extract_blocks, pdf_files
from .text_utils import normalize, read_jsonl

CATEGORIES = {"single_fact", "multi_part", "multi_hop", "cross_document", UNANSWERABLE}
DERIVED = {"multi_hop", "cross_document"}      # gold facts may be computed, not literally in the text


def load_docs():
    """{pdf file name: [normalized block texts]}"""
    return {f.name: [normalize(b["text"]) for b in extract_blocks(f)] for f in pdf_files(PDF_DIR)}


def validate(rows, docs):
    errors, warnings = [], []
    seen = set()

    if len(rows) > MAX_QUESTIONS:
        errors.append(f"dataset has {len(rows)} questions, limit is {MAX_QUESTIONS}")

    for r in rows:
        rid = r.get("id", "?")
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)

        cat = r.get("category")
        if cat not in CATEGORIES:
            errors.append(f"{rid}: bad category {cat!r}")

        sources = r.get("source")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{rid}: `source` must be a non-empty list of PDF file names")
            continue
        missing = [s for s in sources if s not in docs]
        if missing:
            errors.append(f"{rid}: source file(s) not found in {PDF_DIR}: {missing}")
            continue
        if cat == "cross_document" and len(sources) < 2:
            errors.append(f"{rid}: cross_document question needs at least 2 sources")
        if cat in ("single_fact", "multi_part") and len(sources) != 1:
            errors.append(f"{rid}: {cat} question should have exactly 1 source (use cross_document otherwise)")

        if cat == UNANSWERABLE:
            if r.get("evidence") or r.get("gold_facts"):
                errors.append(f"{rid}: unanswerable question must have no evidence/gold_facts")
            continue

        if not r.get("evidence"):
            errors.append(f"{rid}: missing evidence")
        if not r.get("gold_facts"):
            errors.append(f"{rid}: missing gold_facts")

        allowed = {s: docs[s] for s in sources}
        full = " ".join(" ".join(b) for b in allowed.values())
        covered = set()
        for ev in r.get("evidence", []):
            e = normalize(ev)
            home = [s for s, blocks in allowed.items() if any(e in b for b in blocks)]
            if home:
                covered.update(home)
                continue
            elsewhere = [s for s, blocks in docs.items() if s not in allowed and any(e in b for b in blocks)]
            if elsewhere:
                errors.append(f"{rid}: evidence is in {elsewhere}, which is not listed in `source`: {ev!r}")
            else:
                where = "spans two PDF blocks" if e in full else "NOT FOUND in the PDFs"
                errors.append(f"{rid}: evidence {where}: {ev!r}")
        if cat == "cross_document" and covered != set(sources):
            errors.append(f"{rid}: cross_document evidence only covers {sorted(covered)}, needs every source {sources}")

        for alts in r.get("gold_facts", []):
            if not any(normalize(a) in full for a in alts):
                (warnings if cat in DERIVED else errors).append(
                    f"{rid}: gold fact not in the source text (typo or derived fact?): {alts}")

    groups = {}
    for r in rows:
        if r.get("paraphrase_group"):
            groups.setdefault(r["paraphrase_group"], []).append(r["id"])
    for g, ids in groups.items():
        if len(ids) < 2:
            errors.append(f"paraphrase group {g!r} has a single member: {ids}")
    return errors, warnings


def main():
    docs = load_docs()
    print("PDFs: " + ", ".join(f"{n} ({len(b)} blocks)" for n, b in docs.items()))

    digests, bad = {}, False
    for v in VERSIONS:
        p = paths(v)["dataset"]
        if not p.exists():
            print(f"[{v}] no dataset at {p}")
            bad = True
            continue
        rows = read_jsonl(p)
        digests[v] = hashlib.sha1(p.read_bytes()).hexdigest()[:10]
        errors, warnings = validate(rows, docs)
        cats, per_doc = {}, {}
        for r in rows:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
            key = " + ".join(r.get("source") or ["?"])
            per_doc[key] = per_doc.get(key, 0) + 1
        print(f"[{v}] {len(rows)} questions {cats}\n     by source {per_doc} | {len(errors)} errors, {len(warnings)} warnings")
        for m in errors:
            print("   ERROR  ", m)
        for m in warnings:
            print("   warning", m)
        bad |= bool(errors)

    if len(set(digests.values())) > 1:
        print(f"\nERROR: datasets differ between versions {digests}. All versions must use identical questions.")
        bad = True
    elif digests:
        print(f"\nAll version datasets identical (sha1 {next(iter(digests.values()))}).")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
