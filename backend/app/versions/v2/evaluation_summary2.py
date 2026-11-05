# import json
# from pathlib import Path
# from statistics import mean

# base_dir = Path(__file__).resolve().parent
# input_file = base_dir / "evaluated_results2.jsonl"

# with open(input_file, "r", encoding="utf-8") as file:
#     results = [
#         json.loads(line)
#         for line in file
#         if line.strip()
#     ]


# metrics = [
#     "answer_relevancy",
#     "faithfulness",
#     "context_relevancy",
#     "correctness"
# ]

# threshold = 0.7

# print("\n========== RAG Evaluation Summary (v2, k=1) ==========\n")

# if not results:
#     print("No evaluation results found.")
#     raise SystemExit


# overall_scores = []

# for metric in metrics:

#     # Ignore missing or None scores
#     scores = [
#         result[metric]
#         for result in results
#         if result.get(metric) is not None
#     ]

#     if not scores:
#         print(metric.replace("_", " ").title())
#         print("No valid scores available.\n")
#         continue

#     average_score = mean(scores)

#     passed = sum(score >= threshold for score in scores)
#     failed = len(scores) - passed

#     pass_rate = (passed / len(scores)) * 100

#     overall_scores.extend(scores)

#     print(metric.replace("_", " ").title())
#     print(f"Average Score : {average_score:.3f}")
#     print(f"Passed        : {passed}/{len(scores)}")
#     print(f"Failed        : {failed}/{len(scores)}")
#     print(f"Pass Rate     : {pass_rate:.2f}%")
#     print(f"Missing Scores: {len(results) - len(scores)}")
#     print("-" * 45)


# # Overall evaluation
# if overall_scores:
#     print("\n========== Overall Performance ==========")

#     print(
#         f"Overall Average Score: "
#         f"{mean(overall_scores):.3f}"
#     )

#     print(f"Total Test Cases: {len(results)}")
#     print(f"Total Valid Scores: {len(overall_scores)}")

# print("\n==========================================")


# Print the full report.  python -m app.versions.v2.evaluation_summary2
from app.versions.toolkit.experiment import summary

if __name__ == "__main__":
    summary("v2")
