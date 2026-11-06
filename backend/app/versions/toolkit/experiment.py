"""summary + save_experiment for any version."""
import json

from .config import paths
from .migrate import archive_single_doc
from .report import build_report, print_report


def summary(version: str):
    p = paths(version)
    if not (p["results"].exists() or p["evaluated"].exists() or p["retrieval"].exists()):
        raise SystemExit(f"No result files found in {p['dir']}. Run the runner / retrieval_eval / eval_all_metrics first.")
    print_report(build_report(version))


def save(version: str):
    archive_single_doc(version)           # keeps the old single-resume experiment file as *.single_doc.json
    p = paths(version)
    rep = build_report(version)

    with open(p["experiment"], "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"Experiment saved to: {p['experiment']}")
    if not rep["judge"]:
        print("Note: no judge scores yet. Judge-free metrics were saved; run eval_all_metrics and save again.")
