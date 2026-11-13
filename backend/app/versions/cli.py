"""One command line for everything. Run from backend/:

  python -m app.versions.cli validate                 check the dataset against the PDF
  python -m app.versions.cli check                    smoke-test the Gemini embedding API
  python -m app.versions.cli index v1                 chunk + embed + upload to Qdrant (v1, v2 or v3)
  python -m app.versions.cli run v1                   answer all questions (resumes if interrupted)
  python -m app.versions.cli judge v1                 DeepEval judge (resumes; use --start/--end for batches)
  python -m app.versions.cli summary v1               print the 10 metrics
  python -m app.versions.cli save v1                  write experiment_v1.json
  python -m app.versions.cli compare                  side by side table of all saved versions
"""
import argparse
import hashlib
import importlib
import json


def cmd_validate(_):
    import pymupdf
    from .toolkit.config import PDF_DIR, paths
    from .toolkit.corpus import pdf_files
    from .toolkit.text_utils import normalize, read_jsonl
    rows = read_jsonl(paths("v1")["dataset"])
    pages = {}
    for f in pdf_files(PDF_DIR):
        for i, pg in enumerate(pymupdf.open(f), start=1):
            pages[(f.name, i)] = normalize(pg.get_text())
    errors, ids = 0, set()
    for r in rows:
        if r["id"] in ids:
            print(f"ERROR duplicate id {r['id']}")
            errors += 1
        ids.add(r["id"])
        if r["category"] != "unanswerable" and not r["evidence"]:
            print(f"ERROR {r['id']}: answerable question without evidence")
            errors += 1
        for e in r["evidence"]:
            src = r["source"][0]
            if normalize(e["text"]) not in pages.get((src, e["page"]), ""):
                print(f"ERROR {r['id']}: evidence not found on page {e['page']}: {e['text'][:70]}")
                errors += 1
    from collections import Counter
    others = sorted({k[0] for k in pages} - {r['source'][0] for r in rows})
    if others:
        print(f"WARNING: other PDFs in {PDF_DIR} will be indexed too and can distract retrieval: {', '.join(others)}")
    print(f"{len(rows)} questions | {dict(Counter(r['category'] for r in rows))} | {errors} errors")
    raise SystemExit(1 if errors else 0)


def cmd_check(_):
    from .toolkit.config import VERSIONS
    from .toolkit.embeddings import get_embedder
    for v, spec in VERSIONS.items():
        e = get_embedder(spec["embedding"]["model"], spec["embedding"]["dim"])
        q = e.embed_query("How many days of casual leave?")
        d = e.embed_documents(["Casual Leave (CL) 8 days", "Notice period for senior employees is 60 days"])
        sim = lambda a, b: sum(x * y for x, y in zip(a, b))
        print(f"{v}: {spec['embedding']['model']} -> {len(q)} dims (mode: {e.mode}) | "
              f"sim(query, casual leave)={sim(q, d[0]):.3f} > sim(query, notice)={sim(q, d[1]):.3f}: {sim(q, d[0]) > sim(q, d[1])}")


def cmd_index(a):
    from .toolkit.config import agent_module
    importlib.import_module(agent_module(a.version).replace(".agent", ".indexing"))


def cmd_run(a):
    import time
    from .toolkit.config import UNANSWERABLE, VERSIONS, paths
    from .toolkit.retry import with_retry
    from .toolkit.text_utils import read_jsonl, write_jsonl
    from .toolkit.config import agent_module
    p = paths(a.version)
    dataset = read_jsonl(p["dataset"])
    if a.limit:
        dataset = dataset[:a.limit]
    sig = hashlib.sha1(json.dumps({k: VERSIONS[a.version][k] for k in ("chunking", "embedding", "retrieval")}, sort_keys=True).encode()).hexdigest()[:10]
    done = {r["id"]: r for r in read_jsonl(p["results"]) if r.get("config_sig") == sig and not r.get("error")}
    agent = importlib.import_module(agent_module(a.version))
    rows = []
    print(f"{a.version}: {len(dataset)} questions ({len(done)} already done)")
    for i, case in enumerate(dataset, 1):
        if case["id"] in done:
            rows.append(done[case["id"]])
            continue
        print(f"[{i}/{len(dataset)}] {case['id']} ({case['category']}) {case['question'][:70]}")
        row = {"id": case["id"], "category": case["category"], "question": case["question"],
               "expected_answer": case["expected_answer"], "config_sig": sig}
        try:
            out = with_retry(lambda: agent.ask(case["question"]), "generate")
            row.update({"actual_answer": out["answer"], "retrieved_chunks": out["retrieved_chunks"], "retrieved_pages": out["retrieved_pages"],
                        "retrieval_ms": out["retrieval_ms"], "ttft_ms": out["ttft_ms"], "total_ms": out["total_ms"],
                        "retrieval_detail": out["retrieval_detail"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"]})
        except SystemExit:
            write_jsonl(p["results"], rows)
            raise
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"  ERROR {row['error']}")
        rows.append(row)
        write_jsonl(p["results"], rows)
        time.sleep(a.sleep)
    write_jsonl(p["results"], rows)
    print(f"Saved {len(rows)} rows to {p['results']}. Next: judge {a.version}")


def cmd_judge(a):
    from .toolkit.judge import run_judge
    run_judge(a.version, a.start, a.end, [m for m in a.metrics.split(",") if m] or None, a.sleep, a.every)


def cmd_summary(a):
    from .toolkit import report
    report.print_summary(report.build(a.version))


def cmd_save(a):
    from .toolkit import report
    report.save(a.version)


def cmd_compare(_):
    from .toolkit import report
    report.compare()


def main():
    ap = argparse.ArgumentParser(prog="python -m app.versions.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("compare").set_defaults(fn=cmd_compare)
    for name, fn in (("index", cmd_index), ("run", cmd_run), ("judge", cmd_judge), ("summary", cmd_summary), ("save", cmd_save)):
        s = sub.add_parser(name)
        s.add_argument("version", choices=["v1", "v2", "v3"])
        s.set_defaults(fn=fn)
        if name == "run":
            s.add_argument("--limit", type=int, default=0, help="only the first N questions (for a quick test)")
            s.add_argument("--sleep", type=float, default=1.0)
        if name == "judge":
            s.add_argument("--start", type=int, default=1)
            s.add_argument("--end", type=int, default=None)
            s.add_argument("--metrics", default="")
            s.add_argument("--sleep", type=float, default=10)
            s.add_argument("--every", type=int, default=1, help="judge only every Nth question (e.g. 2 = half the cost)")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
