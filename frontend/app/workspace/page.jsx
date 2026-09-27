"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import { Button, ErrorBanner, Spinner } from "@/components/ui";
import CorpusBanner from "@/components/CorpusBanner";
import { api } from "@/lib/api";

export default function UploadPage() {
  const [documents, setDocuments] = useState({});
  const [testDataset, setTestDataset] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  // New test-dataset row being built
  const [newQuestion, setNewQuestion] = useState("");
  const [newRelevantDocs, setNewRelevantDocs] = useState([]);

  const docIds = Object.keys(documents);

  async function refresh() {
    try {
      const [docs, dataset] = await Promise.all([api.getDocuments(), api.getTestDataset()]);
      setDocuments(docs);
      setTestDataset(dataset);
    } catch (e) {
      // Backend probably not running yet — that's fine on first load.
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleFileUpload(e) {
    if (!e.target.files || e.target.files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      await api.uploadDocuments(Array.from(e.target.files));
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function handleTestDatasetFileUpload(e) {
    if (!e.target.files || e.target.files.length === 0) return;
    setError(null);
    try {
      await api.uploadTestDatasetFile(e.target.files[0]);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      e.target.value = "";
    }
  }

  function toggleRelevantDoc(docId) {
    setNewRelevantDocs((prev) => (prev.includes(docId) ? prev.filter((d) => d !== docId) : [...prev, docId]));
  }

  async function addQuestion() {
    if (!newQuestion.trim() || newRelevantDocs.length === 0) return;
    const item = {
      id: `q${testDataset.length + 1}`,
      question: newQuestion.trim(),
      relevant_doc_ids: newRelevantDocs,
    };
    const updated = [...testDataset, item];
    setError(null);
    try {
      await api.saveTestDataset(updated);
      setTestDataset(updated);
      setNewQuestion("");
      setNewRelevantDocs([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function removeQuestion(id) {
    const updated = testDataset.filter((item) => item.id !== id);
    try {
      if (updated.length > 0) {
        await api.saveTestDataset(updated);
      }
      setTestDataset(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">1. Upload your knowledge base</h1>
        <p className="mt-1 text-sm text-slate-500">
          The workspace opens with a benchmarked demo corpus and its labeled test set. Upload your own document(s)
          below only if you want to try the system on different content — that replaces the current corpus, and
          you can always jump back with "Reset to demo corpus".
        </p>
      </div>

      <CorpusBanner variant="full" onReset={refresh} />

      {error && <ErrorBanner message={error} />}

      <Card title="Documents">
        <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 px-6 py-10 text-center hover:border-brand-500">
          <span className="text-sm font-medium text-slate-700">Click to choose files, or drag them here</span>
          <span className="mt-1 text-xs text-slate-400">
            .txt, .md, .pdf, .docx, .csv, .html, .json, and other text-based formats — multiple files supported
          </span>
          <input type="file" multiple className="hidden" onChange={handleFileUpload} />
        </label>

        {uploading && (
          <div className="mt-4">
            <Spinner />
          </div>
        )}

        {docIds.length > 0 && (
          <div className="mt-6">
            <h3 className="mb-2 text-sm font-medium text-slate-700">
              Current corpus ({docIds.length} document{docIds.length !== 1 && "s"})
            </h3>
            <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {docIds.map((id) => (
                <li key={id} className="truncate rounded-md bg-slate-100 px-3 py-2 text-sm text-slate-700">
                  {id}
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>

      <div>
        <h1 className="text-2xl font-bold text-slate-900">2. Build a test dataset</h1>
        <p className="mt-1 text-sm text-slate-500">
          Add questions and mark which uploaded document(s) contain the answer. This is what Evaluate and
          Experiment score against.
        </p>
      </div>

      <Card title="Add a question">
        {docIds.length === 0 ? (
          <p className="text-sm text-slate-400">Upload documents first.</p>
        ) : (
          <div className="space-y-4">
            <input
              type="text"
              placeholder="e.g. How many weeks of parental leave do employees get?"
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
              value={newQuestion}
              onChange={(e) => setNewQuestion(e.target.value)}
            />
            <div>
              <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                Which document(s) answer this?
              </div>
              <div className="flex flex-wrap gap-2">
                {docIds.map((id) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => toggleRelevantDoc(id)}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                      newRelevantDocs.includes(id)
                        ? "bg-brand-600 text-white"
                        : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    }`}
                  >
                    {id}
                  </button>
                ))}
              </div>
            </div>
            <Button onClick={addQuestion} disabled={!newQuestion.trim() || newRelevantDocs.length === 0}>
              Add question
            </Button>
          </div>
        )}

        <div className="mt-6 border-t border-slate-200 pt-4">
          <label className="text-sm font-medium text-slate-700">
            Or upload an existing test_dataset.json:{" "}
            <input type="file" accept=".json" className="mt-2 block text-sm" onChange={handleTestDatasetFileUpload} />
          </label>
        </div>
      </Card>

      {testDataset.length > 0 && (
        <Card title={`Current test dataset (${testDataset.length} question${testDataset.length !== 1 ? "s" : ""})`}>
          <ul className="divide-y divide-slate-100">
            {testDataset.map((item) => (
              <li key={item.id} className="flex items-start justify-between gap-4 py-3">
                <div>
                  <p className="text-sm font-medium text-slate-800">{item.question}</p>
                  <p className="mt-1 text-xs text-slate-400">Relevant: {item.relevant_doc_ids.join(", ")}</p>
                </div>
                <button
                  onClick={() => removeQuestion(item.id)}
                  className="shrink-0 text-xs font-medium text-red-500 hover:text-red-700"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {docIds.length > 0 && testDataset.length > 0 && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
          You're ready — head to <strong>Playground</strong> to ask questions, or <strong>Evaluate</strong> /{" "}
          <strong>Experiment</strong> to score this setup.
        </div>
      )}
    </div>
  );
}