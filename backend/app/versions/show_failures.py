# Show questions that scored low, with expected vs actual answers.
# Usage (from backend/):
#   python -m app.versions.show_failures v3             # correctness < 0.7
#   python -m app.versions.show_failures v3 0.95        # custom threshold
#   python -m app.versions.show_failures v3 0.7 completeness   # use another metric
import json
import sys
from pathlib import Path

version = sys.argv[1] if len(sys.argv) > 1 else "v3"
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
metric = sys.argv[3] if len(sys.argv) > 3 else "correctness"
n = version[1:]

path = Path(__file__).resolve().parent / version / f"evaluated_results{n}.jsonl"
with open(path, "r", encoding="utf-8") as file:
    rows = [json.loads(line) for line in file if line.strip()]

failed = [r for r in rows if r.get(metric) is not None and r[metric] < threshold]
print(f"\n{version}: {len(failed)}/{len(rows)} judged questions with {metric} < {threshold}\n")

for r in sorted(failed, key=lambda r: r[metric]):
    print("=" * 70)
    scores = " | ".join(f"{m} {r[m]:.2f}" for m in
                        ("correctness", "completeness", "faithfulness", "context_relevancy") if r.get(m) is not None)
    print(f"{r['id']} ({r['category']}, {r.get('source')}) | {scores}")
    print(f"Q:        {r['question']}")
    print(f"EXPECTED: {r['expected_answer']}")
    print(f"ACTUAL:   {r['actual_answer']}")
