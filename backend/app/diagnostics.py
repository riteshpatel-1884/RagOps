"""
Phase 4: Diagnostic Engine — "Why is my RAG bad?"

Takes the metrics Phase 1 (retrieval) and Phase 2 (generation) already
compute — recall, precision, mrr, ndcg, faithfulness, relevance — and turns
a bare number like "Recall@5 = 0.45" into an actual diagnosis: which stage
is the bottleneck, why, and what to try next.

This module is pure rule-based reasoning over metrics that already exist —
no new evaluators, no LLM calls. It's meant to sit on top of a single
Phase 1/2 run (`diagnose_config`) or an entire Phase 3 experiment sweep
(`diagnose_experiment_results` + `pareto_frontier`).

IMPORTANT — these thresholds are heuristics, not universal truths.
They were chosen to be reasonable defaults for a small (~10-50 question)
evaluation set, but the right thresholds genuinely depend on your domain,
dataset size, and risk tolerance. Treat them as a starting point to tune,
not a fixed spec — that tuning is itself worth mentioning in an interview.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from pathlib import Path

# --- Tunable thresholds -----------------------------------------------------
LOW_RECALL = 0.5
HIGH_RECALL = 0.7
LOW_PRECISION = 0.35
HIGH_PRECISION = 0.55
LOW_MRR = 0.75
NDCG_RECALL_GAP = 0.15          # ndcg meaningfully below recall -> ranking problem
LOW_FAITHFULNESS = 0.85
LOW_RELEVANCE = 0.45


@dataclass
class Diagnosis:
    bottleneck: str
    evidence: List[str] = field(default_factory=list)
    likely_causes: List[str] = field(default_factory=list)
    suggested_experiments: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "bottleneck": self.bottleneck,
            "evidence": self.evidence,
            "likely_causes": self.likely_causes,
            "suggested_experiments": self.suggested_experiments,
        }

    def print(self):
        print(f"  Bottleneck: {self.bottleneck}")
        if self.evidence:
            print("  Evidence:")
            for e in self.evidence:
                print(f"    • {e}")
        if self.likely_causes:
            print("  Likely causes:")
            for c in self.likely_causes:
                print(f"    → {c}")
        if self.suggested_experiments:
            print("  Suggested next experiments:")
            for i, s in enumerate(self.suggested_experiments, start=1):
                print(f"    {i}. {s['label']}")


def diagnose_config(result: dict) -> Diagnosis:
    """
    Diagnose a single pipeline config's aggregate metrics (one row from
    evaluate.py, evaluate_generation.py, or one entry of an experiment.py
    results file). Multiple rules can fire — a config can have more than
    one bottleneck at once.
    """
    recall = result.get("recall")
    precision = result.get("precision")
    mrr = result.get("mrr")
    ndcg = result.get("ndcg")
    faithfulness = result.get("faithfulness")
    relevance = result.get("relevance")

    evidence: List[str] = []
    causes: List[str] = []
    suggestions: List[dict] = []
    bottlenecks: List[str] = []

    # --- Rule 1: low recall + high precision -> top_k too low ---
    if recall is not None and precision is not None and recall < LOW_RECALL and precision >= HIGH_PRECISION:
        bottlenecks.append("Retrieval recall")
        evidence.append(f"Recall is low ({recall:.2f}) but precision is high ({precision:.2f}) — "
                         f"what's retrieved is relevant, there just isn't enough of it.")
        causes.append("top_k is too low for this chunk size.")
        suggestions.append({
            "label": "Increase top_k",
            "param": "top_k", "op": "add", "value": 2,
        })

    # --- Rule 2: high recall + low precision -> chunks too large / top_k too high ---
    if recall is not None and precision is not None and recall >= HIGH_RECALL and precision < LOW_PRECISION:
        bottlenecks.append("Retrieval precision")
        evidence.append(f"Recall is high ({recall:.2f}) but precision is low ({precision:.2f}) — "
                         f"most relevant chunks are being retrieved, but so is a lot of noise.")
        causes.append("chunk_size is too large (chunks span multiple topics) or top_k is too high.")
        suggestions.append({
            "label": "Decrease chunk_size",
            "param": "chunk_size", "op": "scale", "value": 0.6,
        })
        suggestions.append({
            "label": "Decrease top_k",
            "param": "top_k", "op": "add", "value": -2,
        })

    # --- Rule 3: low MRR -> relevant chunk retrieved but ranked poorly ---
    if mrr is not None and mrr < LOW_MRR:
        bottlenecks.append("Retrieval ranking")
        evidence.append(f"MRR is low ({mrr:.2f}) — the first relevant chunk often isn't ranked near the top.")
        causes.append("Embedding model isn't distinguishing relevant from irrelevant chunks well, "
                       "or a reranking stage is missing.")
        suggestions.append({
            "label": "Try a stronger embedding model (bge-m3)",
            "param": "embedder_name", "op": "set", "value": "bge-m3",
        })

    # --- Rule 4: ndcg well below recall -> ranking quality problem despite ok recall ---
    if recall is not None and ndcg is not None and (recall - ndcg) > NDCG_RECALL_GAP:
        bottlenecks.append("Retrieval ranking")
        evidence.append(f"NDCG ({ndcg:.2f}) is notably below recall ({recall:.2f}) — relevant chunks are "
                         f"being retrieved, just not ranked near the top of the list.")
        causes.append("No reranking stage; the embedding model's similarity ordering is coarse.")
        suggestions.append({
            "label": "Add a reranking stage (not yet built — candidate for Phase 5+)",
            "param": None, "op": None, "value": None,
        })

    # --- Rule 5: decent recall but low faithfulness -> generation problem, not retrieval ---
    if faithfulness is not None and recall is not None and faithfulness < LOW_FAITHFULNESS and recall >= LOW_RECALL:
        bottlenecks.append("Generation faithfulness")
        evidence.append(f"Recall is reasonable ({recall:.2f}) but faithfulness is low ({faithfulness:.2f}) — "
                         f"the right context is being retrieved, but the generator isn't using it faithfully.")
        causes.append("Weak generator model, loose prompt instructions, or context window crowded with "
                       "irrelevant chunks distracting the model.")
        suggestions.append({
            "label": "Try a stronger/different generator model",
            "param": "generator_name", "op": "set", "value": "groq",
        })
        suggestions.append({
            "label": "Decrease top_k so less (noisier) context reaches the generator",
            "param": "top_k", "op": "add", "value": -2,
        })

    # --- Rule 6: both recall and faithfulness low -> fix retrieval first ---
    if faithfulness is not None and recall is not None and faithfulness < LOW_FAITHFULNESS and recall < LOW_RECALL:
        bottlenecks.append("Retrieval (upstream of generation)")
        evidence.append(f"Both recall ({recall:.2f}) and faithfulness ({faithfulness:.2f}) are low.")
        causes.append("Generation can't be faithful to context that isn't there in the first place — "
                       "retrieval is the root cause here, not the generator.")
        suggestions.append({
            "label": "Fix retrieval first (increase top_k or improve embedder) before tuning generation",
            "param": "top_k", "op": "add", "value": 2,
        })

    # --- Rule 7: low relevance despite good faithfulness -> answer may be too narrow/off-target ---
    if relevance is not None and faithfulness is not None and relevance < LOW_RELEVANCE and faithfulness >= LOW_FAITHFULNESS:
        bottlenecks.append("Answer relevance")
        evidence.append(f"Faithfulness is fine ({faithfulness:.2f}) but relevance is low ({relevance:.2f}) — "
                         f"the answer is accurate to the context but may not be squarely addressing the question.")
        causes.append("Prompt doesn't emphasize directly answering the question, or a weaker embedding "
                       "model is producing a less discriminative relevance score.")
        suggestions.append({
            "label": "Re-check with a stronger embedder (bge-m3) before trusting this relevance score",
            "param": "embedder_name", "op": "set", "value": "bge-m3",
        })

    if not bottlenecks:
        return Diagnosis(
            bottleneck="None detected",
            evidence=["All available metrics are within normal ranges for this rule set."],
            likely_causes=[],
            suggested_experiments=[],
        )

    return Diagnosis(
        bottleneck=" + ".join(dict.fromkeys(bottlenecks)),  # dedupe, keep order
        evidence=evidence,
        likely_causes=causes,
        suggested_experiments=suggestions,
    )


def apply_suggestion(config_dict: dict, suggestion: dict):
    """
    Turn a suggested_experiments entry into a new PipelineConfig by applying
    its param change to an existing config dict. Returns None if the
    suggestion has no actionable param (e.g. "add a reranker" — not built yet).
    """
    from pipeline import PipelineConfig  # deferred import to avoid a hard dependency for pure diagnosis use

    if not suggestion.get("param"):
        return None

    new_values = dict(config_dict)
    param, op, value = suggestion["param"], suggestion["op"], suggestion["value"]

    if op == "add":
        new_values[param] = max(1, new_values[param] + value)
    elif op == "scale":
        new_values[param] = max(20, int(new_values[param] * value))
    elif op == "set":
        new_values[param] = value

    # PipelineConfig fields only — drop anything extraneous.
    allowed = {"chunk_size", "chunk_overlap", "embedder_name", "top_k", "collection_name", "generator_name"}
    return PipelineConfig(**{k: v for k, v in new_values.items() if k in allowed})


# ---------------------------------------------------------------------------
# Multi-config analysis (operates on a Phase 3 experiment_results list)
# ---------------------------------------------------------------------------

def pareto_frontier(results: List[dict], metric_x: str, metric_y: str) -> List[dict]:
    """
    Return the configs where no other config is at least as good on BOTH
    metrics and strictly better on at least one — i.e. no other config
    dominates them. These are the only configs worth considering; every
    other config is strictly worse on at least one axis for no benefit on
    the other.
    """
    candidates = [r for r in results if metric_x in r and metric_y in r]
    frontier = []
    for r in candidates:
        dominated = any(
            other is not r
            and other[metric_x] >= r[metric_x]
            and other[metric_y] >= r[metric_y]
            and (other[metric_x] > r[metric_x] or other[metric_y] > r[metric_y])
            for other in candidates
        )
        if not dominated:
            frontier.append(r)
    return sorted(frontier, key=lambda r: r[metric_x], reverse=True)


def describe_tradeoff(a: dict, b: dict, metric_x: str, metric_y: str) -> str:
    """Plain-English description of the trade-off between two configs on two metrics."""
    dx = a[metric_x] - b[metric_x]
    dy = a[metric_y] - b[metric_y]
    ca, cb = a["config"], b["config"]

    def fmt(c):
        return f"chunk={c['chunk_size']}/{c['chunk_overlap']}, top_k={c['top_k']}, embedder={c['embedder_name']}, gen={c['generator_name']}"

    return (
        f"Config A ({fmt(ca)}) vs Config B ({fmt(cb)}): "
        f"A has {dx:+.3f} {metric_x} and {dy:+.3f} {metric_y} relative to B."
    )


def build_diagnostic_report(results: List[dict], metric_x: str = "recall", metric_y: str = "faithfulness") -> dict:
    """
    Structured (JSON-friendly) version of the full diagnostic report — no
    printing. Used by both the CLI (diagnose_experiment_results, which
    prints this) and the API server (which returns it directly as JSON).
    """
    per_config = []
    for r in results:
        per_config.append({"config": r["config"], "metrics": {k: v for k, v in r.items() if k != "config"},
                            "diagnosis": diagnose_config(r).as_dict()})

    report = {"per_config": per_config, "metric_x": metric_x, "metric_y": metric_y}

    if metric_y not in results[0]:
        report["frontier"] = None
        report["tradeoff"] = f"Skipped: '{metric_y}' not present in these results (sweep likely used skip_generation)."
        return report

    frontier = pareto_frontier(results, metric_x, metric_y)
    report["frontier"] = frontier

    if len(frontier) >= 2:
        best_x = max(frontier, key=lambda r: r[metric_x])
        best_y = max(frontier, key=lambda r: r[metric_y])
        report["tradeoff"] = (
            None if best_x is best_y
            else describe_tradeoff(best_x, best_y, metric_x, metric_y)
        )
    else:
        report["tradeoff"] = None

    return report


def diagnose_experiment_results(results: List[dict], metric_x: str = "recall", metric_y: str = "faithfulness"):
    """Print a full diagnostic report over an entire Phase 3 experiment sweep."""
    report = build_diagnostic_report(results, metric_x, metric_y)

    print(f"\n=== Per-config diagnosis ({len(results)} configs) ===")
    for i, entry in enumerate(report["per_config"], start=1):
        c = entry["config"]
        print(f"\n[{i}] chunk={c['chunk_size']}/{c['chunk_overlap']}  top_k={c['top_k']}  "
              f"embedder={c['embedder_name']}  generator={c['generator_name']}")
        d = entry["diagnosis"]
        print(f"  Bottleneck: {d['bottleneck']}")
        if d["evidence"]:
            print("  Evidence:")
            for e in d["evidence"]:
                print(f"    • {e}")
        if d["likely_causes"]:
            print("  Likely causes:")
            for c_ in d["likely_causes"]:
                print(f"    → {c_}")
        if d["suggested_experiments"]:
            print("  Suggested next experiments:")
            for i2, s in enumerate(d["suggested_experiments"], start=1):
                print(f"    {i2}. {s['label']}")

    if report["frontier"] is None:
        print(f"\n{report['tradeoff']}")
        return

    print(f"\n=== Pareto frontier: {metric_x} vs {metric_y} ===")
    print(f"(Configs where no other config is at least as good on both metrics — everything")
    print(f" else in this sweep is dominated and not worth considering on these two axes.)\n")
    for r in report["frontier"]:
        c = r["config"]
        print(f"  chunk={c['chunk_size']}/{c['chunk_overlap']}  top_k={c['top_k']}  "
              f"embedder={c['embedder_name']}  generator={c['generator_name']}  "
              f"-- {metric_x}={r[metric_x]:.3f}, {metric_y}={r[metric_y]:.3f}")

    if report["tradeoff"]:
        print(f"\n=== Trade-off between frontier extremes ===")
        print(f"  {report['tradeoff']}")
    elif len(report["frontier"]) >= 2:
        print(f"\n  One config leads on both {metric_x} and {metric_y} — no real trade-off in this sweep.")


def latest_results_file(results_dir: Path) -> Optional[Path]:
    files = sorted(results_dir.glob("experiment_*.json"))
    return files[-1] if files else None