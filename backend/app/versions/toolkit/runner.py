"""Generate answers for every dataset question with a version's RAG pipeline.

Resumable: finished questions are kept, failed ones are retried. Use --force to regenerate all.
Saved after EVERY question, so an interrupted run never leaves an empty file.
"""
import argparse
import importlib
import json
import shutil
import time

from .config import VERSIONS, paths, rag_module
from .retry import with_retry
from .text_utils import read_jsonl, write_jsonl

DATASET_FIELDS = ("category", "paraphrase_group", "expected_answer", "gold_facts", "evidence")


def _is_complete(row, case):
    return (
        row is not None
        and row.get("question") == case["question"]
        and row.get("error") is None
        and row.get("actual_answer") is not None
        and "retrieved_chunks" in row            # rows from the old runner lack this -> regenerate
    )


def main(version: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate every answer")
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds between questions")
    args = parser.parse_args()

    p = paths(version)
    ask = importlib.import_module(rag_module(version, "agent")).ask
    cases = read_jsonl(p["dataset"])
    if not cases:
        raise SystemExit(f"Dataset missing or empty: {p['dataset']}\nCopy eval_dataset.jsonl into that folder.")

    existing_rows = read_jsonl(p["results"])
    if existing_rows and any("retrieved_chunks" not in r for r in existing_rows):
        legacy = p["results"].with_name(p["results"].stem + ".legacy.jsonl")
        if not legacy.exists():
            shutil.copy(p["results"], legacy)
            print(f"Old-format results backed up to {legacy.name}")

    existing = {} if args.force else {r.get("id"): r for r in existing_rows if r.get("id")}
    done = {}

    def save():
        write_jsonl(p["results"], [done[c["id"]] for c in cases if c["id"] in done])

    reused = 0
    for i, case in enumerate(cases, 1):
        prev = existing.get(case["id"])
        if _is_complete(prev, case):
            done[case["id"]] = {**prev, **{k: case[k] for k in DATASET_FIELDS}}
            reused += 1
            continue

        print(f"\n[{i}/{len(cases)}] ({case['category']}) {case['question']}")
        base = {"id": case["id"], "question": case["question"], **{k: case[k] for k in DATASET_FIELDS}}

        try:
            out = with_retry(lambda: ask(case["question"]), "ask")
            row = {**base, "actual_answer": out["answer"], "context": out["context"],
                   "retrieved_chunks": out["retrieved_chunks"], "retriever_k": out["retriever_k"],
                   "retrieval_ms": out["retrieval_ms"], "generation_ms": out["generation_ms"],
                   "total_ms": out["total_ms"], "input_tokens": out["input_tokens"],
                   "output_tokens": out["output_tokens"], "error": None}
            print(f"Answer: {out['answer']}")
            print(f"  retrieval {out['retrieval_ms']:.0f} ms | generation {out['generation_ms']:.0f} ms | "
                  f"tokens in/out {out['input_tokens']}/{out['output_tokens']}")
        except SystemExit:
            save()
            raise
        except Exception as e:
            print(f"Error: {e}")
            row = {**base, "actual_answer": None, "context": None, "retrieved_chunks": [], "retriever_k": None,
                   "retrieval_ms": None, "generation_ms": None, "total_ms": None,
                   "input_tokens": 0, "output_tokens": 0, "error": str(e)}

        done[case["id"]] = row
        save()                       # progress is never lost
        time.sleep(args.sleep)       # eases free-tier rate limits

    save()
    errors = sum(1 for r in done.values() if r.get("error"))
    print(f"\nDone. {len(done)}/{len(cases)} rows saved to {p['results'].name} "
          f"({reused} reused, {errors} with errors - re-run the same command to retry those).")
