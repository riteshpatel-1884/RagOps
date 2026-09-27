"use client";

import { useEffect, useState } from "react";
import { RotateCcw } from "lucide-react";
import { api } from "@/lib/api";

// Shown at the top of Upload / Evaluate / Experiment. Tells the viewer
// (often an interviewer trying their own document) whether they're looking
// at the benchmarked demo corpus or a document uploaded at runtime, and
// gives them one click back to the benchmarked corpus.
//
// `variant`:
//   "full"  — Upload page: shows doc/question counts + reset button.
//   "warn"  — Evaluate/Experiment: only renders when source is "custom",
//             since that's the case where metrics may not mean much
//             (no labeled test set was written for this corpus).
export default function CorpusBanner({ variant = "full", onReset }) {
  const [status, setStatus] = useState(null);
  const [resetting, setResetting] = useState(false);

  async function refresh() {
    try {
      setStatus(await api.getCorpusStatus());
    } catch {
      // backend not up yet — fine on first load
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleReset() {
    setResetting(true);
    try {
      await api.resetDemoCorpus();
      await refresh();
      onReset?.();
    } catch {
      // surface nothing here — caller pages already have their own error banners
    } finally {
      setResetting(false);
    }
  }

  if (!status) return null;
  if (variant === "warn" && status.source !== "custom") return null;

  const isDemo = status.source === "demo";

  return (
    <div
      className={`flex flex-wrap items-center justify-between gap-3 rounded-lg border px-4 py-3 text-sm ${
        isDemo
          ? "border-border bg-surface text-muted"
          : "border-amber-300/40 bg-amber-500/10 text-amber-700"
      }`}
    >
      <div>
        {isDemo ? (
          <span>
            Showing the <strong>benchmarked demo corpus</strong> ({status.doc_count} document
            {status.doc_count !== 1 ? "s" : ""}, {status.test_dataset_count} labeled question
            {status.test_dataset_count !== 1 ? "s" : ""}).
          </span>
        ) : variant === "full" ? (
          <span>
            Showing a <strong>custom uploaded corpus</strong> ({status.doc_count} document
            {status.doc_count !== 1 ? "s" : ""}). Evaluate/Experiment metrics only mean something if you've
            added labeled questions for it above.
          </span>
        ) : (
          <span>
            You're evaluating a <strong>custom uploaded corpus</strong>, not the benchmarked demo corpus — these
            metrics only reflect whatever test-set questions were added for it.
          </span>
        )}
      </div>

      {status.has_seed_demo && !isDemo && (
        <button
          type="button"
          onClick={handleReset}
          disabled={resetting}
          className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text transition-colors hover:bg-surfaceHover disabled:opacity-50"
        >
          <RotateCcw size={13} />
          {resetting ? "Resetting…" : "Reset to demo corpus"}
        </button>
      )}
    </div>
  );
}