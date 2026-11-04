import json
import time
from pathlib import Path

from app.versions.v2.RAG_2.agent import ask

V2_DIR = Path(__file__).resolve().parent
DATASET = V2_DIR / "eval_dataset.jsonl"   # v2's own dataset copy
OUTPUT = V2_DIR / "results2.jsonl"        # v2's own results

with open(DATASET, "r", encoding="utf-8") as f:
    test_cases = [json.loads(line) for line in f if line.strip()]

with open(OUTPUT, "w", encoding="utf-8") as out:
    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        print(f"\n[{i}/{len(test_cases)}] Question: {question}")

        try:
            rag_result = ask(question)
            actual_answer = rag_result["answer"]
            context = rag_result["context"]
            error = None
        except Exception as e:
            actual_answer, context, error = None, [], str(e)
            print(f"Error: {e}")

        result = {
            "question": question,
            "expected_answer": case["expected_answer"],
            "actual_answer": actual_answer,
            "context": context,
            "error": error,
        }

        # Write each result immediately so a crash doesn't lose progress
        out.write(json.dumps(result, ensure_ascii=False) + "\n")
        out.flush()

        print(f"Answer: {actual_answer}")
        time.sleep(1)  # eases free-tier rate limits

print(f"\nDone! Results saved to {OUTPUT.name}")