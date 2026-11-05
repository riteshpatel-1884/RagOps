# import json
# from pathlib import Path
# from statistics import mean

# RETRIEVER_K = 3
# EXPERIMENT_NAME = "experiment_v3_k3_merged"

# base_dir = Path(__file__).resolve().parent

# input_file = base_dir / "evaluated_results3.jsonl"
# output_file = base_dir / "experiment_v3.json"

# with open(input_file, "r", encoding="utf-8") as file:
#     results = [
#         json.loads(line)
#         for line in file
#         if line.strip()
#     ]


# def avg(metric: str):
#     scores = [r[metric] for r in results if r.get(metric) is not None]
#     return mean(scores) if scores else None


# experiment = {
#     "experiment": EXPERIMENT_NAME,
#     "retriever_k": RETRIEVER_K,
#     "number_of_test_cases": len(results),

#     "metrics": {
#         "answer_relevancy": avg("answer_relevancy"),
#         "faithfulness": avg("faithfulness"),
#         "context_relevancy": avg("context_relevancy"),
#         "correctness": avg("correctness"),
#     }
# }

# with open(output_file, "w", encoding="utf-8") as file:
#     json.dump(experiment, file, indent=4)

# print(f"Experiment saved to: {output_file}")


# Write experiment_v3.json.  python -m app.versions.v3.save_experiment3
from app.versions.toolkit.experiment import save

if __name__ == "__main__":
    save("v3")
