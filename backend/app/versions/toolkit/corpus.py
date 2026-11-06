"""Load every PDF in backend/pdfs/ into text blocks that remember WHERE they came from.

A block/chunk is a dict:  {"text": str, "source": "file.pdf", "page": 1, "pages": [1]}
`source` is the PDF file name (it is what the dataset's `source` field refers to).
"""
import re
from pathlib import Path
from typing import List

from .config import PDF_DIR


def pdf_files(pdf_dir: Path = PDF_DIR) -> List[Path]:
    files = sorted(Path(pdf_dir).glob("*.pdf"), key=lambda p: p.name.lower())
    if not files:
        raise SystemExit(f"No PDFs found in {pdf_dir}. Put ritesh_AI.pdf and NEXORA_TECHNOLOGIES_PVT.pdf there.")
    return files


def extract_blocks(pdf_path) -> List[dict]:
    """Text blocks of one PDF, top to bottom, with the page number (same extraction as before)."""
    import pymupdf
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    blocks = []
    for page_no, page in enumerate(doc, 1):
        raw = sorted([b for b in page.get_text("blocks") if b[6] == 0], key=lambda b: (b[1], b[0]))
        for b in raw:
            text = re.sub(r"\s*\n\s*", " ", b[4]).strip()
            if text:
                blocks.append({"text": text, "source": pdf_path.name, "page": page_no, "pages": [page_no]})
    doc.close()
    return blocks


def load_blocks(pdf_dir: Path = PDF_DIR) -> List[dict]:
    """Blocks of every PDF (alphabetical by file name, each document in reading order)."""
    out: List[dict] = []
    for f in pdf_files(pdf_dir):
        out.extend(extract_blocks(f))
    return out


def merge_blocks(blocks: List[dict], max_chars: int = 700, overlap: int = 1) -> List[dict]:
    """Merge consecutive blocks into bigger chunks so a heading stays with its details.

    Never merges across documents. The last `overlap` blocks are repeated at the start of the next chunk.
    """
    def emit(group):
        pages = sorted({pg for b in group for pg in b["pages"]})
        return {"text": " ".join(b["text"] for b in group), "source": group[0]["source"],
                "page": pages[0], "pages": pages}

    chunks: List[dict] = []
    for source in dict.fromkeys(b["source"] for b in blocks):          # keeps document order
        current: List[dict] = []
        for b in (x for x in blocks if x["source"] == source):
            size = sum(len(x["text"]) for x in current) + len(current)
            if current and size + len(b["text"]) > max_chars:
                chunks.append(emit(current))
                current = current[-overlap:] if overlap else []
            current.append(b)
        if current:
            chunks.append(emit(current))
    return chunks


def describe(chunks: List[dict]) -> str:
    per = {}
    for c in chunks:
        per[c["source"]] = per.get(c["source"], 0) + 1
    return ", ".join(f"{s}: {n}" for s, n in per.items())


if __name__ == "__main__":
    blocks = load_blocks()
    merged = merge_blocks(blocks)
    print(f"blocks   ({len(blocks)}): {describe(blocks)}")
    print(f"merged   ({len(merged)}): {describe(merged)}")
