# # Slow, sequential evaluation with batching, resume and retry on Groq limits.
# #
# # Usage (from backend/):
# #   python -m app.versions.v1.eval_all_metrics1                       # all tests
# #   python -m app.versions.v1.eval_all_metrics1 --start 1 --end 7     # batch 1
# #   python -m app.versions.v1.eval_all_metrics1 --start 8 --end 15    # batch 2
# #
# # Already-evaluated tests are skipped, so it is safe to re-run after any failure.
# import argparse
# import json
# import re
# import time
# from pathlib import Path
# from dotenv import load_dotenv

# from deepeval.metrics import (
#     AnswerRelevancyMetric,
#     FaithfulnessMetric,
#     ContextualRelevancyMetric,
#     GEval
# )

# from deepeval.test_case import LLMTestCase, LLMTestCaseParams
# from deepeval.models import DeepEvalBaseLLM
# from langchain_groq import ChatGroq

# load_dotenv()

# METRIC_KEYS = [
#     "answer_relevancy",
#     "faithfulness",
#     "context_relevancy",
#     "correctness",
# ]

# SLEEP_BETWEEN_TESTS = 15   # seconds, lets the tokens-per-minute window recover
# MAX_RETRIES = 8
# LONG_WAIT_LIMIT = 120      # if Groq asks to wait longer than this (sec), stop and resume later


# # ==========================================
# # 1. Custom Groq Evaluation Model
# # ==========================================

# class GroqEvalModel(DeepEvalBaseLLM):

#     def __init__(self):
#         self.model = ChatGroq(
#             model="openai/gpt-oss-120b",
#             temperature=0
#         )

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str, schema=None):

#         if schema:
#             structured_model = self.model.with_structured_output(
#                 schema,
#                 method="json_mode"
#             )
#             return structured_model.invoke(prompt)

#         response = self.model.invoke(prompt)
#         return response.content

#     async def a_generate(self, prompt: str, schema=None):

#         if schema:
#             structured_model = self.model.with_structured_output(
#                 schema,
#                 method="json_mode"
#             )
#             return await structured_model.ainvoke(prompt)

#         response = await self.model.ainvoke(prompt)
#         return response.content

#     def get_model_name(self):
#         return "Groq GPT-OSS-120B"


# eval_model = GroqEvalModel()


# # ==========================================
# # 2. Retry helper (rate limit + connection errors)
# # ==========================================

# def parse_wait_seconds(msg: str):
#     """Parse 'try again in 14m22.5s' / '5.5s' / '340ms' from Groq's message."""
#     m = re.search(r"try again in ([0-9hms.]+)", msg)
#     if not m:
#         return None

#     total = 0.0
#     for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
#         v = float(value)
#         total += {"h": v * 3600, "m": v * 60, "s": v, "ms": v / 1000}[unit]

#     return total


# def with_retry(fn, label):
#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             return fn()
#         except Exception as e:
#             msg = str(e)

#             if "429" in msg or "rate_limit" in msg.lower():
#                 wait = parse_wait_seconds(msg)

#                 if wait is not None and wait > LONG_WAIT_LIMIT:
#                     print(f"\nLong rate limit: Groq asks to wait ~{wait / 60:.0f} min.")
#                     print("Progress is saved. Re-run the same command after that time.")
#                     raise SystemExit(1)

#                 wait = (wait + 3) if wait is not None else 20
#                 reason = "rate limit"

#             elif "Connection error" in msg or "ReadError" in msg or "timed out" in msg.lower():
#                 wait = 10
#                 reason = "connection error"

#             else:
#                 raise  # unknown error, do not hide it

#             print(f"  [{label}] {reason}, retry {attempt}/{MAX_RETRIES} in {wait:.0f}s")
#             time.sleep(wait)

#     raise RuntimeError(f"{label} failed after {MAX_RETRIES} retries")


# # ==========================================
# # 3. Args and paths (all inside v1)
# # ==========================================

# parser = argparse.ArgumentParser()
# parser.add_argument("--start", type=int, default=1, help="first test number (1-based)")
# parser.add_argument("--end", type=int, default=None, help="last test number (inclusive)")
# args = parser.parse_args()

# base_dir = Path(__file__).resolve().parent

# input_file = base_dir / "results1.jsonl"
# output_file = base_dir / "evaluated_results1.jsonl"


# # ==========================================
# # 4. Load results + any already evaluated
# # ==========================================

# with open(input_file, "r", encoding="utf-8") as file:
#     results = [json.loads(line) for line in file if line.strip()]

# if not results:
#     raise SystemExit(
#         "results1.jsonl is empty. Run: python -m app.versions.v1.eval_runner1 first."
#     )

# done = {}   # question -> evaluated row
# if output_file.exists():
#     with open(output_file, "r", encoding="utf-8") as file:
#         for line in file:
#             if line.strip():
#                 row = json.loads(line)
#                 if all(row.get(k) is not None for k in METRIC_KEYS):
#                     done[row["question"]] = row


# def save_all():
#     """Write evaluated rows in original dataset order."""
#     with open(output_file, "w", encoding="utf-8") as file:
#         for r in results:
#             if r["question"] in done:
#                 file.write(json.dumps(done[r["question"]], ensure_ascii=False) + "\n")


# end = min(args.end or len(results), len(results))
# print(f"Total tests: {len(results)} | running {args.start}-{end} | already done: {len(done)}")


# # ==========================================
# # 5. Evaluate the selected slice
# # ==========================================

# for i in range(args.start, end + 1):

#     result = results[i - 1]

#     if result["question"] in done:
#         print(f"\nTest {i}/{len(results)}: already evaluated, skipping")
#         continue

#     print(f"\nEvaluating test {i}/{len(results)}")

#     context = result["context"]
#     retrieval_context = [context] if isinstance(context, str) else context

#     test_case = LLMTestCase(
#         input=result["question"],
#         actual_output=result["actual_answer"],
#         expected_output=result["expected_answer"],
#         retrieval_context=retrieval_context
#     )

#     answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=eval_model)
#     faithfulness = FaithfulnessMetric(threshold=0.7, model=eval_model)
#     context_relevancy = ContextualRelevancyMetric(threshold=0.7, model=eval_model)
#     correctness = GEval(
#         name="Correctness",
#         criteria=(
#             "Determine whether the actual output is factually correct "
#             "compared with the expected output. Evaluate factual accuracy, "
#             "missing information, and incorrect claims."
#         ),
#         evaluation_params=[
#             LLMTestCaseParams.ACTUAL_OUTPUT,
#             LLMTestCaseParams.EXPECTED_OUTPUT
#         ],
#         threshold=0.7,
#         model=eval_model
#     )

#     with_retry(lambda: answer_relevancy.measure(test_case), "answer_relevancy")
#     with_retry(lambda: faithfulness.measure(test_case), "faithfulness")
#     with_retry(lambda: context_relevancy.measure(test_case), "context_relevancy")
#     with_retry(lambda: correctness.measure(test_case), "correctness")

#     done[result["question"]] = {
#         **result,

#         "answer_relevancy": answer_relevancy.score,
#         "faithfulness": faithfulness.score,
#         "context_relevancy": context_relevancy.score,
#         "correctness": correctness.score,

#         "answer_relevancy_passed": answer_relevancy.is_successful(),
#         "faithfulness_passed": faithfulness.is_successful(),
#         "context_relevancy_passed": context_relevancy.is_successful(),
#         "correctness_passed": correctness.is_successful()
#     }

#     # Save after every test so a crash never loses progress
#     save_all()

#     print("Answer Relevancy:", answer_relevancy.score)
#     print("Faithfulness:", faithfulness.score)
#     print("Context Relevancy:", context_relevancy.score)
#     print("Correctness:", correctness.score)

#     time.sleep(SLEEP_BETWEEN_TESTS)


# # ==========================================
# # 6. Final status
# # ==========================================

# print("\n================================")
# print(f"Evaluated so far: {len(done)}/{len(results)}")
# print(f"Results Saved: {output_file}")
# if len(done) == len(results):
#     print("All tests done. Now run evaluation_summary1 and save_experiment1.")
# else:
#     print("Some tests are still pending. Run the next batch.")
# print("================================")


# LLM-judge metrics (resumable, batchable).  python -m app.versions.v1.eval_all_metrics1 [--start 1 --end 20]
from app.versions.toolkit.judge_eval import main

if __name__ == "__main__":
    main("v1")
