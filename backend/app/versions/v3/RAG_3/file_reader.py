import re
from pathlib import Path
from typing import List

import pymupdf


def extract_paragraphs(pdf_path: str | Path) -> List[str]:
    """Extract paragraphs from a PDF and return them as a list of strings."""
    doc = pymupdf.open(pdf_path)
    paragraphs = []

    for page in doc:
        # Only text blocks (type 0), skipping image blocks
        blocks = [b for b in page.get_text("blocks") if b[6] == 0]

        # Sort top to bottom, then left to right
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        for block in blocks:
            # Replace line breaks inside a paragraph with spaces
            text = re.sub(r"\s*\n\s*", " ", block[4]).strip()
            if text:
                paragraphs.append(text)

    doc.close()
    return paragraphs


def merge_paragraphs(
    paragraphs: List[str],
    max_chars: int = 700,
    overlap: int = 1,
) -> List[str]:
    """Merge consecutive blocks into larger chunks so a heading stays with its
    details. The last `overlap` blocks are repeated at the start of the next chunk."""
    chunks: List[str] = []
    current: List[str] = []

    for p in paragraphs:
        size = sum(len(x) for x in current) + len(current)  # chars + joining spaces
        if current and size + len(p) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.append(p)

    if current:
        chunks.append(" ".join(current))

    return chunks


if __name__ == "__main__":
    pdf = Path(__file__).resolve().parents[4] / "pdfs" / "ritesh_AI.pdf"
    paras = extract_paragraphs(pdf)
    merged = merge_paragraphs(paras)
    print(f"{len(paras)} blocks -> {len(merged)} merged chunks")
    for i, c in enumerate(merged, 1):
        print(f"\n--- chunk {i} ({len(c)} chars) ---\n{c}")
