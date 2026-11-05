# import json
# import time
# from pathlib import Path

# from app.versions.v1.RAG_1.agent import ask

# V1_DIR = Path(__file__).resolve().parent
# DATASET = V1_DIR / "eval_dataset.jsonl"   # v1's own dataset copy
# OUTPUT = V1_DIR / "results1.jsonl"        # v1's own results

# with open(DATASET, "r", encoding="utf-8") as f:
#     test_cases = [json.loads(line) for line in f if line.strip()]

# with open(OUTPUT, "w", encoding="utf-8") as out:
#     for i, case in enumerate(test_cases, 1):
#         question = case["question"]
#         print(f"\n[{i}/{len(test_cases)}] Question: {question}")

#         try:
#             rag_result = ask(question)
#             actual_answer = rag_result["answer"]
#             context = rag_result["context"]
#             error = None
#         except Exception as e:
#             actual_answer, context, error = None, [], str(e)
#             print(f"Error: {e}")

#         result = {
#             "question": question,
#             "expected_answer": case["expected_answer"],
#             "actual_answer": actual_answer,
#             "context": context,
#             "error": error,
#         }

#         # Write each result immediately so a crash doesn't lose progress
#         out.write(json.dumps(result, ensure_ascii=False) + "\n")
#         out.flush()

#         print(f"Answer: {actual_answer}")
#         time.sleep(1)  # eases free-tier rate limits

# print(f"\nDone! Results saved to {OUTPUT.name}")


# Generate answers for all dataset questions (resumable).  python -m app.versions.v1.eval_runner1 [--force]
from app.versions.toolkit.runner import main

if __name__ == "__main__":
    main("v1")
