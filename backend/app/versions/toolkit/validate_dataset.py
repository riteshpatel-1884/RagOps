"""Check the evaluation dataset against the PDF and across versions.

Usage (from backend/):
    python -m app.versions.toolkit.validate_dataset

Catches: evidence/gold facts that are not in the PDF (e.g. after you edit your resume),
evidence spanning two PDF blocks (would be unfair to v1/v2), duplicate ids, and
datasets that differ between version folders (versions must use the SAME questions).
"""
import hashlib
import re
import sys

from .config import PDF_PATH, UNANSWERABLE, VERSIONS, paths
from .text_utils import normalize, read_jsonl

CATEGORIES = {"single_fact", "multi_part", "multi_hop", UNANSWERABLE}


def pdf_blocks(pdf_path):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    blocks = []
    for page in doc:
        bl = sorted([b for b in page.get_text("blocks") if b[6] == 0], key=lambda b: (b[1], b[0]))
        for b in bl:
            t = re.sub(r"\s*\n\s*", " ", b[4]).strip()
            if t:
                blocks.append(t)
    doc.close()
    return blocks


def validate(rows, blocks):
    errors, warnings = [], []
    norm_blocks = [normalize(b) for b in blocks]
    full = " ".join(norm_blocks)
    seen = set()

    for r in rows:
        rid = r.get("id", "?")
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)

        if r.get("category") not in CATEGORIES:
            errors.append(f"{rid}: bad category {r.get('category')!r}")

        if r.get("category") == UNANSWERABLE:
            if r.get("evidence") or r.get("gold_facts"):
                errors.append(f"{rid}: unanswerable question must have no evidence/gold_facts")
            continue

        if not r.get("evidence"):
            errors.append(f"{rid}: missing evidence")
        if not r.get("gold_facts"):
            errors.append(f"{rid}: missing gold_facts")

        for ev in r.get("evidence", []):
            e = normalize(ev)
            if not any(e in b for b in norm_blocks):
                where = "spans two PDF blocks" if e in full else "NOT FOUND in the PDF"
                errors.append(f"{rid}: evidence {where}: {ev!r}")

        for alts in r.get("gold_facts", []):
            if not any(normalize(a) in full for a in alts):
                (warnings if r.get("category") == "multi_hop" else errors).append(
                    f"{rid}: gold fact not in PDF text (typo or derived fact?): {alts}")

    groups = {}
    for r in rows:
        if r.get("paraphrase_group"):
            groups.setdefault(r["paraphrase_group"], []).append(r["id"])
    for g, ids in groups.items():
        if len(ids) < 2:
            errors.append(f"paraphrase group {g!r} has a single member: {ids}")
    return errors, warnings


def main():
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")
    blocks = pdf_blocks(PDF_PATH)
    print(f"PDF: {PDF_PATH.name} ({len(blocks)} text blocks)")

    digests = {}
    bad = False
    for v in VERSIONS:
        p = paths(v)["dataset"]
        if not p.exists():
            print(f"[{v}] no dataset at {p}")
            continue
        rows = read_jsonl(p)
        digests[v] = hashlib.sha1(p.read_bytes()).hexdigest()[:10]
        errors, warnings = validate(rows, blocks)
        cats = {}
        for r in rows:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        print(f"[{v}] {len(rows)} questions {cats} | {len(errors)} errors, {len(warnings)} warnings")
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
