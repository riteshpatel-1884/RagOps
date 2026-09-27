"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import { Button, ErrorBanner, Spinner } from "@/components/ui";
import { LabeledMultiNumberInput, LabeledSelect, parseNumberList } from "@/components/FormControls";
import CorpusBanner from "@/components/CorpusBanner";
import { api } from "@/lib/api";

const METRIC_OPTIONS = ["recall", "precision", "mrr", "ndcg", "faithfulness", "relevance", "elapsed_seconds"];

export default function ExperimentPage() {
  const [options, setOptions] = useState(null);

  const [chunkSizesStr, setChunkSizesStr] = useState("200,400");
  const [chunkOverlapsStr, setChunkOverlapsStr] = useState("40,80");
  const [topKsStr, setTopKsStr] = useState("3,5");
  const [embedder, setEmbedder] = useState("bge-small-en-v1.5");
  const [generator, setGenerator] = useState("groq");
  const [judge, setJudge] = useState("offline");
  const [skipGeneration, setSkipGeneration] = useState(false);
  const [sortBy, setSortBy] = useState("recall");

  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.configOptions().then(setOptions).catch(() => {});
  }, []);

  async function runSweep() {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const res = await api.runExperiment({
        chunk_sizes: parseNumberList(chunkSizesStr),
        chunk_overlaps: parseNumberList(chunkOverlapsStr),
        top_ks: parseNumberList(topKsStr),
        embedders: [embedder],
        generators: [generator],
        judge,
        skip_generation: skipGeneration,
        sort_by: sortBy,
      });
      setResponse(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const metricKeys = response
    ? Object.keys(response.results[0] || {}).filter((k) => k !== "config" && k !== "error")
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Experiment</h1>
        <p className="mt-1 text-sm text-slate-500">
          Sweep a grid of chunk sizes, overlaps, and top-K values, score every combination, and rank them.
        </p>
      </div>

      <CorpusBanner variant="warn" />

      {error && <ErrorBanner message={error} />}

      <Card title="Sweep config">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <LabeledMultiNumberInput label="Chunk sizes" value={chunkSizesStr} onChange={setChunkSizesStr} />
          <LabeledMultiNumberInput label="Chunk overlaps" value={chunkOverlapsStr} onChange={setChunkOverlapsStr} />
          <LabeledMultiNumberInput label="Top-K values" value={topKsStr} onChange={setTopKsStr} />
          <LabeledSelect
            label="Embedder"
            value={embedder}
            onChange={setEmbedder}
            options={options?.embedders || [{ value: "bge-small-en-v1.5", label: "BGE Small EN v1.5" }]}
          />
          <LabeledSelect
            label="Generator"
            value={generator}
            onChange={setGenerator}
            options={options?.generators || [{ value: "groq", label: "Groq" }]}
          />
          <LabeledSelect label="Rank by" value={sortBy} onChange={setSortBy} options={METRIC_OPTIONS.map((m) => ({ value: m, label: m }))} />
        </div>

        <div className="mt-4 flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            <input
              type="checkbox"
              checked={skipGeneration}
              onChange={(e) => setSkipGeneration(e.target.checked)}
              className="rounded border-slate-300"
            />
            Skip generation (retrieval metrics only — much faster)
          </label>

          {!skipGeneration && (
            <div className="w-48">
              <LabeledSelect
                label="Judge"
                value={judge}
                onChange={setJudge}
                options={options?.judges || [{ value: "offline", label: "Offline" }]}
              />
            </div>
          )}
        </div>

        <div className="mt-4">
          <Button onClick={runSweep} disabled={loading}>
            Run sweep
          </Button>
        </div>
        {loading && (
          <div className="mt-4">
            <Spinner />
          </div>
        )}
        {!skipGeneration && (
          <p className="mt-2 text-xs text-amber-600">
            Real LLM generators make one API call per question per config — a wide sweep can be slow and costly.
          </p>
        )}
      </Card>

      {response && (
        <Card title={`Results (${response.results.length} config${response.results.length !== 1 ? "s" : ""}, ranked by ${response.sort_by})`}>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-xs uppercase text-slate-500">
                  <th className="py-2 pr-4">#</th>
                  <th className="py-2 pr-4">Chunk</th>
                  <th className="py-2 pr-4">Overlap</th>
                  <th className="py-2 pr-4">Top K</th>
                  <th className="py-2 pr-4">Embedder</th>
                  <th className="py-2 pr-4">Generator</th>
                  {metricKeys.map((m) => (
                    <th key={m} className="py-2 pr-4">
                      {m}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {response.results.map((row, i) => (
                  <tr key={i} className={`border-b border-slate-100 ${i === 0 ? "bg-brand-50" : ""}`}>
                    <td className="py-2 pr-4 font-medium text-slate-500">{i + 1}</td>
                    <td className="py-2 pr-4">{row.config.chunk_size}</td>
                    <td className="py-2 pr-4">{row.config.chunk_overlap}</td>
                    <td className="py-2 pr-4">{row.config.top_k}</td>
                    <td className="py-2 pr-4">{row.config.embedder_name}</td>
                    <td className="py-2 pr-4">{row.config.generator_name}</td>
                    {metricKeys.map((m) => (
                      <td key={m} className="py-2 pr-4 text-slate-700">
                        {typeof row[m] === "number" ? row[m].toFixed(3) : "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {response.failed.length > 0 && (
            <p className="mt-4 text-xs text-red-500">{response.failed.length} config(s) failed and were excluded.</p>
          )}
          <p className="mt-4 text-xs text-slate-400">
            Saved as <code>{response.filename}</code> — open it to compare against other sweeps.
          </p>
        </Card>
      )}
    </div>
  );
}