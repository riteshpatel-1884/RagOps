import json
import re
import unicodedata
from pathlib import Path

_DASHES = {ord(c): " " for c in "‐‑‒–—―−-"}
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}


def normalize(text) -> str:
    """Casefold, unify dashes/quotes and whitespace so 'Text-to-SQL' == 'text to sql'."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).casefold()
    t = t.translate(_DASHES).translate(_QUOTES)
    return re.sub(r"\s+", " ", t).strip()


def tokens(text) -> list:
    return re.findall(r"\w+", normalize(text))


def read_jsonl(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
