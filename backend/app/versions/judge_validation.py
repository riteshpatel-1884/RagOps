# Check how far you can trust the LLM judge: label a sample BLIND, then compare with the judge.
#
#   python -m app.versions.judge_validation export --n 20
#       -> writes judge_validation_labels.csv (no judge scores shown) and judge_validation_key.json
#   Open the CSV and fill the human_score column: 1 = fully correct, 0.5 = partly correct, 0 = wrong.
#   python -m app.versions.judge_validation report
#       -> correlation, pass/fail agreement, Cohen's kappa, mean absolute difference
import argparse
import csv
import json
import random
from pathlib import Path
from statistics import correlation, mean

from app.versions.toolkit.config import JUDGE_THRESHOLD, VERSIONS, paths
from app.versions.toolkit.text_utils import read_jsonl

HERE = Path(__file__).resolve().parent
LABELS = HERE / "judge_validation_labels.csv"
KEY = HERE / "judge_validation_key.json"


def export(n, seed):
    pool = []
    for v in VERSIONS:
        for r in read_jsonl(paths(v)["evaluated"]):
            if r.get("correctness") is not None:
                pool.append({**r, "version": v})
    if not pool:
        raise SystemExit("No judged answers found. Run eval_all_metrics for at least one version first.")
    random.Random(seed).shuffle(pool)
    sample = pool[:n]

    with open(LABELS, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "question", "expected_answer", "actual_answer", "human_score"])
        for i, r in enumerate(sample, 1):
            w.writerow([i, r["question"], r["expected_answer"], r["actual_answer"], ""])
    KEY.write_text(json.dumps({str(i): {"version": r["version"], "id": r["id"], "judge": r["correctness"]}
                               for i, r in enumerate(sample, 1)}, indent=2), encoding="utf-8")
    print(f"Wrote {LABELS.name} ({len(sample)} rows). Fill 'human_score' (1 / 0.5 / 0), then run: report")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in (True, False))
    return None if pe == 1 else (po - pe) / (1 - pe)


def report():
    if not LABELS.exists() or not KEY.exists():
        raise SystemExit("Run `export` first.")
    key = json.loads(KEY.read_text(encoding="utf-8"))
    pairs = []
    with open(LABELS, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            h = (row.get("human_score") or "").strip().replace(",", ".")
            if h:
                pairs.append((float(h), key[row["sample_id"]]["judge"]))
    if len(pairs) < 5:
        raise SystemExit(f"Only {len(pairs)} labelled rows. Label at least 5 (ideally 20+).")

    human, judge = [p[0] for p in pairs], [p[1] for p in pairs]
    hp, jp = [h >= JUDGE_THRESHOLD for h in human], [j >= JUDGE_THRESHOLD for j in judge]
    print(f"\nLabelled answers: {len(pairs)}")
    if len(set(human)) > 1 and len(set(judge)) > 1:
        print(f"Pearson correlation (human vs judge): {correlation(human, judge):.3f}")
    print(f"Mean absolute difference:             {mean(abs(h - j) for h, j in pairs):.3f}")
    print(f"Pass/fail agreement @ {JUDGE_THRESHOLD}:           {sum(a == b for a, b in zip(hp, jp)) / len(pairs):.0%}")
    k = kappa(hp, jp)
    print(f"Cohen's kappa (pass/fail):            {'n/a' if k is None else f'{k:.3f}'}")
    print(f"Judge stricter than you: {sum(j < h for h, j in pairs)} | more lenient: {sum(j > h for h, j in pairs)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "report"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    export(a.n, a.seed) if a.mode == "export" else report()
