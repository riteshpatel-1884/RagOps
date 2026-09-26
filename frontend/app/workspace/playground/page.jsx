"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import { Button, ErrorBanner, Spinner } from "@/components/ui";
import { LabeledInput, LabeledSelect } from "@/components/FormControls";
import { api } from "@/lib/api";

export default function PlaygroundPage() {
  const [question, setQuestion] = useState("");
  const [chunkSize, setChunkSize] = useState(400);
  const [chunkOverlap, setChunkOverlap] = useState(80);
  const [embedder, setEmbedder] = useState("bge-small-en-v1.5");
  const [generator, setGenerator] = useState("groq");
  const [topK, setTopK] = useState(5);

  const [options, setOptions] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.configOptions().then(setOptions).catch(() => {});
  }, []);

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.ask({
        question: question.trim(),
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        embedder_name: embedder,
        generator_name: generator,
        top_k: topK,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Playground</h1>
        <p className="mt-1 text-sm text-slate-500">
          Ask a one-off question against your uploaded corpus — no test dataset required. Good for sanity-checking
          a config before running a full evaluation.
        </p>
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
          <LabeledSelect
            label="Generator"
            value={generator}
            onChange={setGenerator}
            options={options?.generators || [{ value: "groq", label: "Groq" }]}
          />
          <LabeledInput label="Top K" type="number" value={topK} onChange={(v) => setTopK(Number(v))} />
        </div>
      </Card>

      <Card>
        <div className="flex gap-3">
          <input
            type="text"
            placeholder="Ask a question about your documents..."
            className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          />
          <Button onClick={handleAsk} disabled={loading || !question.trim()}>
            Ask
          </Button>
        </div>
        {loading && (
          <div className="mt-4">
            <Spinner />
          </div>
        )}
      </Card>

      {result && (
        <Card title="Answer">
          <p className="whitespace-pre-wrap text-sm text-slate-800">{result.answer}</p>

          <div className="mt-6">
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
              Retrieved chunks ({result.retrieved_hits.length})
            </h3>
            <ul className="space-y-2">
              {result.retrieved_hits.map((hit, i) => (
                <li key={hit.chunk_id} className="rounded-md border border-slate-200 p-3 text-sm">
                  <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
                    <span>
                      #{i + 1} · {hit.doc_id}
                    </span>
                    <span>score {hit.score.toFixed(3)}</span>
                  </div>
                  <p className="text-slate-700">{hit.text}</p>
                </li>
              ))}
            </ul>
          </div>
        </Card>
      )}
    </div>
  );
}
