# Usage (from backend/):
#   python -m app.versions.compare_experiments              # v1 (baseline) vs v2
#   python -m app.versions.compare_experiments v1 v3        # any two versions
import json
import sys
from pathlib import Path

versions_dir = Path(__file__).resolve().parent

baseline_ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
experiment_ver = sys.argv[2] if len(sys.argv) > 2 else "v2"


def load(version: str) -> dict:
    path = versions_dir / version / f"experiment_{version}.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run save_experiment for {version} first.")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


baseline = load(baseline_ver)
experiment = load(experiment_ver)

base_label = f"{baseline_ver} (k={baseline['retriever_k']})"
exp_label = f"{experiment_ver} (k={experiment['retriever_k']})"

print("\n========== EXPERIMENT COMPARISON ==========\n")
print(f"{'Metric':<22} {base_label:<14} {exp_label:<14} {'Change':<8}")
print("-" * 60)

for metric, base_score in baseline["metrics"].items():

    exp_score = experiment["metrics"].get(metric)

    if base_score is None or exp_score is None:
        print(f"{metric.replace('_', ' ').title():<22} {'n/a':<14} {'n/a':<14} {'n/a':<8}")
        continue

    change = exp_score - base_score

    print(
        f"{metric.replace('_', ' ').title():<22} "
        f"{base_score:<14.3f} "
        f"{exp_score:<14.3f} "
        f"{change:+.3f}"
    )

print()



# (backend) PS D:\ragops\backend>  python -m app.versions.compare_experiments

# ========== EXPERIMENT COMPARISON ==========

# Metric                 v1 (k=3)       v2 (k=1)       Change  
# ------------------------------------------------------------
# Answer Relevancy       0.959          0.967          +0.008
# Faithfulness           0.867          0.933          +0.067
# Context Relevancy      0.849          0.317          -0.532
# Correctness            0.740          0.247          -0.493

# (backend) PS D:\ragops\backend> 