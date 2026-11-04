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


if __name__ == "__main__":
    pdf = Path(__file__).resolve().parents[2] / "pdfs" / "ritesh_AI.pdf"
    print(extract_paragraphs(pdf))