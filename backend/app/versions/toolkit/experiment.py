"""summary + save_experiment for any version."""
import json
import shutil

from .config import paths
from .report import build_report, print_report


def summary(version: str):
    p = paths(version)
    if not (p["results"].exists() or p["evaluated"].exists() or p["retrieval"].exists()):
        raise SystemExit(f"No result files found in {p['dir']}. Run the runner / retrieval_eval / eval_all_metrics first.")
    print_report(build_report(version))


def save(version: str):
    p = paths(version)
    rep = build_report(version)

    if p["experiment"].exists():          # keep the old experiment file (older runs had a different format/PDF)
        try:
            old = json.loads(p["experiment"].read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if "retrieval" not in old:
            legacy = p["experiment"].with_name(p["experiment"].stem + ".legacy.json")
            if not legacy.exists():
                shutil.copy(p["experiment"], legacy)
                print(f"Old experiment file backed up to {legacy.name}")

    with open(p["experiment"], "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"Experiment saved to: {p['experiment']}")
    if not rep["judge"]:
        print("Note: no judge scores yet. Judge-free metrics were saved; run eval_all_metrics and save again.")
