"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import MetricStat from "@/components/MetricStat";
import { Button, ErrorBanner, Spinner } from "@/components/ui";
import { LabeledInput, LabeledSelect } from "@/components/FormControls";
import CorpusBanner from "@/components/CorpusBanner";
import { api } from "@/lib/api";

export default function EvaluatePage() {
  const [tab, setTab] = useState("retrieval");
  const [options, setOptions] = useState(null);

  const [chunkSize, setChunkSize] = useState(400);
  const [chunkOverlap, setChunkOverlap] = useState(80);
  const [embedder, setEmbedder] = useState("bge-small-en-v1.5");
  const [topK, setTopK] = useState(5);
  const [generator, setGenerator] = useState("groq");
  const [judge, setJudge] = useState("offline");

  const [retrievalResult, setRetrievalResult] = useState(null);
  const [generationResult, setGenerationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.configOptions().then(setOptions).catch(() => {});
  }, []);

  async function runRetrieval() {
    setLoading(true);
    setError(null);
    setRetrievalResult(null);
    try {
      const res = await api.evaluateRetrieval({
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        embedder_name: embedder,
        top_k: topK,
      });
      setRetrievalResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function runGeneration() {
    setLoading(true);
    setError(null);
    setGenerationResult(null);
    try {
      const res = await api.evaluateGeneration({
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        embedder_name: embedder,
        top_k: topK,
        generator_name: generator,
        judge,
      });
      setGenerationResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Evaluate</h1>
        <p className="mt-1 text-sm text-slate-500">
          Run one pipeline config against your full test dataset and see retrieval or generation metrics.
        </p>
      </div>

      <CorpusBanner variant="warn" />

      <div className="flex gap-1 border-b border-slate-200">
        {["retrieval", "generation"].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium capitalize ${
              tab === t ? "border-b-2 border-brand-600 text-brand-700" : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {error && <ErrorBanner message={error} />}

      <Card title="Config">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
          <LabeledInput label="Chunk size" type="number" value={chunkSize} onChange={(v) => setChunkSize(Number(v))} />
          <LabeledInput
            label="Chunk overlap"
            type="number"
            value={chunkOverlap}
            onChange={(v) => setChunkOverlap(Number(v))}
          />
          <LabeledSelect
            label="Embedder"
            value={embedder}
            onChange={setEmbedder}
            options={options?.embedders || [{ value: "bge-small-en-v1.5", label: "BGE Small EN v1.5" }]}
          />
          <LabeledInput label="Top K" type="number" value={topK} onChange={(v) => setTopK(Number(v))} />
          {tab === "generation" && (
            <LabeledSelect
              label="Generator"
              value={generator}
              onChange={setGenerator}
              options={options?.generators || [{ value: "groq", label: "Groq" }]}
            />
          )}
        </div>
        {tab === "generation" && (
          <div className="mt-4 w-48">
            <LabeledSelect
              label="Judge"
              value={judge}
              onChange={setJudge}
              options={options?.judges || [{ value: "offline", label: "Offline" }]}
            />
          </div>
        )}
        <div className="mt-4">
          <Button onClick={tab === "retrieval" ? runRetrieval : runGeneration} disabled={loading}>
            Run {tab} evaluation
          </Button>
        </div>
        {loading && (
          <div className="mt-4">
            <Spinner />
          </div>
        )}
      </Card>

      {tab === "retrieval" && retrievalResult && (
        <>
          <Card title="Overall metrics">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {Object.entries(retrievalResult.overall).map(([key, value]) => (
                <MetricStat key={key} label={key} value={value} />
              ))}
            </div>
          </Card>

          <Card title="Per-question results">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-xs uppercase text-slate-500">
                    <th className="py-2 pr-4">Question</th>
                    <th className="py-2 pr-4">Top hit doc</th>
                    <th className="py-2 pr-4">1st rel. rank</th>
                  </tr>
                </thead>
                <tbody>
                  {retrievalResult.per_query.map((row) => (
                    <tr key={row.id} className="border-b border-slate-100">
                      <td className="max-w-md py-2 pr-4 text-slate-700">{row.question}</td>
                      <td className="py-2 pr-4 text-slate-500">{row.top_hit_doc ?? "—"}</td>
                      <td className="py-2 pr-4 text-slate-500">
                        {row.first_relevant_rank === -1 ? "not found" : row.first_relevant_rank}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === "generation" && generationResult && (
        <>
          <Card title="Overall metrics">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricStat label="avg faithfulness" value={generationResult.avg_faithfulness} />
              <MetricStat label="avg relevance" value={generationResult.avg_relevance} />
            </div>
          </Card>

          <Card title="Per-question results">
            <div className="space-y-4">
              {generationResult.rows.map((row) => (
                <div key={row.id} className="rounded-lg border border-slate-200 p-4">
                  <p className="text-sm font-medium text-slate-800">{row.question}</p>
                  <p className="mt-2 text-sm text-slate-600">{row.answer}</p>
                  <div className="mt-3 flex flex-wrap gap-4 text-xs text-slate-500">
                    <span>
                      Faithfulness: <strong className="text-slate-700">{row.faithfulness.toFixed(2)}</strong> —{" "}
                      {row.faithfulness_explanation}
                    </span>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-4 text-xs text-slate-500">
                    <span>
                      Relevance: <strong className="text-slate-700">{row.relevance.toFixed(2)}</strong> —{" "}
                      {row.relevance_explanation}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}