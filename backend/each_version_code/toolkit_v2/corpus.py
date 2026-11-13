"""PDF -> text blocks -> chunks. Three chunking strategies (static, semantic, parent_child).

Add more pages / PDFs: drop them in backend/pdfs/ and re-run `index`. Nothing here is specific to one document.
"""
import re
from collections import Counter
from pathlib import Path
from typing import List

from .config import PDF_DIR

PAGE_NUM = re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.I)


def pdf_files(pdf_dir: Path = PDF_DIR) -> List[Path]:
    files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDF found in {pdf_dir}. Put the handbook PDF there.")
    return files


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def extract_blocks(pdf_path) -> List[dict]:
    """Layout blocks per page, with running headers/footers ('Page 3 of 27', repeated title lines) removed."""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    raw = []
    for pno, page in enumerate(doc, start=1):
        for b in sorted(page.get_text("blocks"), key=lambda b: (round(b[1]), b[0])):
            if b[6] != 0:                                    # skip images
                continue
            text = _clean(b[4])
            if text:
                raw.append({"source": Path(pdf_path).name, "page": pno, "text": text})
    pages = max((r["page"] for r in raw), default=0)
    counts = Counter(r["text"] for r in raw)
    boiler = {t for t, c in counts.items() if pages >= 4 and c >= max(3, pages // 2)}   # repeated on most pages
    blocks, pending_bullet = [], False
    for r in raw:
        if r["text"] in boiler or PAGE_NUM.match(r["text"]):
            continue
        if r["text"] in ("•", "-", "–"):                     # lone bullet glyph: prefix the next block
            pending_bullet = True
            continue
        if pending_bullet:
            r = {**r, "text": "• " + r["text"]}
            pending_bullet = False
        blocks.append(r)
    return blocks


def load_blocks(pdf_dir: Path = PDF_DIR) -> List[dict]:
    out = []
    for f in pdf_files(pdf_dir):
        out.extend(extract_blocks(f))
    return out


def _page_text(blocks) -> List[dict]:
    """One text per (source, page): blocks joined with newlines."""
    pages, order = {}, []
    for b in blocks:
        key = (b["source"], b["page"])
        if key not in pages:
            pages[key] = []
            order.append(key)
        pages[key].append(b["text"])
    return [{"source": s, "page": p, "text": "\n".join(pages[(s, p)])} for s, p in order]


# ---------------------------------------------------------------- static / recursive splitting
def _recursive_split(text: str, size: int, overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True,
                                              separators=["\n\n", "\n", ". ", " ", ""])
    return [(d.page_content, d.metadata["start_index"]) for d in splitter.create_documents([text])]


def _concat_with_offsets(blocks):
    """Whole document as one text (so chunks can cross page breaks) + offsets to map a position back to its page."""
    parts, offsets, pos = [], [], 0
    for pt in _page_text(blocks):
        offsets.append((pos, pt["source"], pt["page"]))
        parts.append(pt["text"])
        pos += len(pt["text"]) + 2
    return "\n\n".join(parts), offsets


def _page_at(offsets, pos):
    cur = offsets[0]
    for o in offsets:
        if o[0] <= pos:
            cur = o
        else:
            break
    return cur[1], cur[2]


def static_chunks(blocks, chunk_chars, overlap_chars):
    text, offsets = _concat_with_offsets(blocks)
    out = []
    for i, (t, start) in enumerate(_recursive_split(text, chunk_chars, overlap_chars)):
        src, page = _page_at(offsets, start)
        out.append({"id": f"c{i:04d}", "text": t.strip(), "source": src, "page": page})
    return out


# ---------------------------------------------------------------- semantic chunking
def _units(blocks, min_unit=150):
    """Small layout blocks are merged into ~sentence-sized units (never across pages)."""
    units, cur, cur_key = [], [], None

    def flush():
        nonlocal cur
        if cur:
            units.append({"source": cur_key[0], "page": cur_key[1], "text": " ".join(cur)})
            cur = []

    for b in blocks:
        key = (b["source"], b["page"])
        if cur and key != cur_key:
            flush()
        cur_key = key
        cur.append(b["text"])
        if sum(len(x) for x in cur) >= min_unit:
            flush()
    flush()
    return units


def _cos_dist(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return 1.0 - dot / (na * nb)


def semantic_chunks(blocks, embedder, percentile, min_chars, max_chars):
    """Split where the meaning shifts: consecutive units are embedded and a new chunk starts
    when their cosine distance is above the given percentile of all distances."""
    units = _units(blocks)
    vecs = embedder.embed_documents([u["text"] for u in units])
    dists = [_cos_dist(vecs[i], vecs[i + 1]) for i in range(len(units) - 1)]
    s = sorted(dists)
    threshold = s[min(len(s) - 1, int(len(s) * percentile / 100))] if s else 1.0

    chunks, cur = [], []

    def close():
        nonlocal cur
        if cur:
            chunks.append({"text": " ".join(u["text"] for u in cur), "source": cur[0]["source"], "page": cur[0]["page"]})
            cur = []

    for i, u in enumerate(units):
        if cur and sum(len(x["text"]) for x in cur) + len(u["text"]) > max_chars:
            close()
        cur.append(u)
        size = sum(len(x["text"]) for x in cur)
        if i < len(dists) and dists[i] > threshold and size >= min_chars:
            close()
    close()
    if len(chunks) > 1 and len(chunks[-1]["text"]) < min_chars:          # fold a tiny tail into the previous chunk
        tail = chunks.pop()
        chunks[-1]["text"] += " " + tail["text"]
    for i, c in enumerate(chunks):
        c["id"] = f"c{i:04d}"
    return chunks


# ---------------------------------------------------------------- parent-child
def parent_child_chunks(blocks, parent_chars, parent_overlap, child_chars, child_overlap):
    """Small CHILD chunks are embedded for accurate matching; each carries its larger PARENT text,
    which is what the LLM finally reads."""
    text, offsets = _concat_with_offsets(blocks)
    out = []
    for pi, (ptext, pstart) in enumerate(_recursive_split(text, parent_chars, parent_overlap)):
        src, page = _page_at(offsets, pstart)
        parent_id = f"p{pi:04d}"
        for ci, (ctext, _) in enumerate(_recursive_split(ptext, child_chars, child_overlap)):
            out.append({"id": f"{parent_id}-c{ci:02d}", "text": ctext.strip(), "source": src, "page": page,
                        "parent_id": parent_id, "parent_text": ptext.strip()})
    return out


def make_chunks(spec: dict, embedder):
    blocks = load_blocks()
    c = spec["chunking"]
    if c["strategy"] == "static":
        return static_chunks(blocks, c["chunk_chars"], c["overlap_chars"])
    if c["strategy"] == "semantic":
        return semantic_chunks(blocks, embedder, c["percentile"], c["min_chars"], c["max_chars"])
    if c["strategy"] == "parent_child":
        return parent_child_chunks(blocks, c["parent_chars"], c["parent_overlap"], c["child_chars"], c["child_overlap"])
    raise ValueError(f"unknown chunking strategy {c['strategy']}")


def describe(chunks) -> str:
    if not chunks:
        return "0 chunks"
    sizes = sorted(len(c["text"]) for c in chunks)
    return f"{len(chunks)} chunks, median {sizes[len(sizes) // 2]} chars, max {sizes[-1]}"
