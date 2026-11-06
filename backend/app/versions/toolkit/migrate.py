"""Move result files of the OLD single-resume experiment out of the way (never deleted).

Old files are renamed  results1.jsonl -> results1.single_doc.jsonl  etc. so the new multi-document
numbers are never mixed with the old ones, and the old ones stay available for comparison.
"""
import json

from .config import paths

# file -> a key that only rows written by the multi-document code contain
MARKERS = {"results": "retrieved_sources", "retrieval": "retrieved_sources", "evaluated": "source"}


def _first_row(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def _archive(path):
    target = path.with_name(f"{path.stem}.single_doc{path.suffix}")
    i = 2
    while target.exists():
        target = path.with_name(f"{path.stem}.single_doc{i}{path.suffix}")
        i += 1
    path.rename(target)
    print(f"[migrate] old single-resume file archived: {path.name} -> {target.name}")


def archive_single_doc(version: str) -> None:
    p = paths(version)
    for name, marker in MARKERS.items():
        f = p[name]
        if f.exists():
            row = _first_row(f)
            if row is not None and marker not in row:
                _archive(f)
    exp = p["experiment"]
    if exp.exists():
        try:
            old = json.loads(exp.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if "corpus" not in old:
            _archive(exp)
