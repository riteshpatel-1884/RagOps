"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import { Button, ErrorBanner, Spinner } from "@/components/ui";
import { LabeledInput, LabeledSelect } from "@/components/FormControls";
import { api } from "@/lib/api";

export default function VersionsPage() {
  const [options, setOptions] = useState(null);
  const [versions, setVersions] = useState([]);
  const [baseline, setBaseline] = useState(null);

  // Record-version form
  const [name, setName] = useState("");
  const [chunkSize, setChunkSize] = useState(400);
  const [chunkOverlap, setChunkOverlap] = useState(80);
  const [embedder, setEmbedder] = useState("bge-small-en-v1.5");
  const [topK, setTopK] = useState(5);
  const [generator, setGenerator] = useState("groq");
  const [judge, setJudge] = useState("offline");
  const [notes, setNotes] = useState("");
  const [setAsBaseline, setSetAsBaseline] = useState(false);

  // Check-regression form
  const [checkName, setCheckName] = useState("");
  const [againstName, setAgainstName] = useState("");
  const [comparison, setComparison] = useState(null);

  const [recording, setRecording] = useState(false);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(null);

  async function refresh() {
    try {
      const [opts, v] = await Promise.all([api.configOptions(), api.listVersions()]);
      setOptions(opts);
      setVersions(v.versions);
      setBaseline(v.baseline);
      if (v.versions.length > 0 && !checkName) setCheckName(v.versions[v.versions.length - 1].name);
    } catch (e) {
      // backend not up yet — fine on first load
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleRecord() {
    if (!name.trim()) return;
    setRecording(true);
    setError(null);
    try {
      await api.recordVersion({
        name: name.trim(),
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        embedder_name: embedder,
        top_k: topK,
        generator_name: generator,
        judge,
        notes,
        set_baseline: setAsBaseline,
      });
      setName("");
      setNotes("");
      setSetAsBaseline(false);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRecording(false);
    }
  }

  async function handleCheck() {
    if (!checkName) return;
    setChecking(true);
    setError(null);
    setComparison(null);
    try {
      const res = await api.checkRegression({
        name: checkName,
        against: againstName || undefined,
      });
      setComparison(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setChecking(false);
    }
  }

  async function handleSetBaseline(versionName) {
    setError(null);
    try {
      await api.setBaseline(versionName);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDelete(versionName) {
    setError(null);
    try {
      await api.deleteVersion(versionName);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Versions</h1>
        <p className="mt-1 text-sm text-slate-500">
          Record a pipeline config as a named version, then check whether a later change is a real improvement or a
          regression — across every metric, not just the one you were targeting.
        </p>
      </div>

      {error && <ErrorBanner message={error} />}

      <Card title="Record a version">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <LabeledInput label="Version name" value={name} onChange={setName} />
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
          <LabeledSelect
            label="Judge"
            value={judge}
            onChange={setJudge}
            options={options?.judges || [{ value: "offline", label: "Offline" }]}
          />
          <LabeledInput label="Notes (optional)" value={notes} onChange={setNotes} />
        </div>

        <label className="mt-4 flex items-center gap-2 text-sm text-slate-600">
          <input
            type="checkbox"
            checked={setAsBaseline}
            onChange={(e) => setSetAsBaseline(e.target.checked)}
            className="rounded border-slate-300"
          />
          Set as baseline
        </label>

        <div className="mt-4">
          <Button onClick={handleRecord} disabled={recording || !name.trim()}>
            Record version
          </Button>
        </div>
        {recording && (
          <div className="mt-4">
            <Spinner />
          </div>
        )}
      </Card>

      <Card title={`Recorded versions (${versions.length})`}>
        {versions.length === 0 ? (
          <p className="text-sm text-slate-400">No versions recorded yet — record one above.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-xs uppercase text-slate-500">
                  <th className="py-2 pr-4">Name</th>
                  <th className="py-2 pr-4">Config</th>
                  <th className="py-2 pr-4">Recall</th>
                  <th className="py-2 pr-4">Faithfulness</th>
                  <th className="py-2 pr-4">Recorded</th>
                  <th className="py-2 pr-4"></th>
                </tr>
              </thead>
              <tbody>
                {versions.map((v) => (
                  <tr key={v.name} className="border-b border-slate-100">
                    <td className="py-2 pr-4 font-medium text-slate-800">
                      {v.name}
                      {v.name === baseline && (
                        <span className="ml-2 rounded-full bg-brand-100 px-2 py-0.5 text-[10px] font-medium text-brand-700">
                          baseline
                        </span>
                      )}
                    </td>
                    <td className="py-2 pr-4 text-xs text-slate-500">
                      chunk={v.config.chunk_size}/{v.config.chunk_overlap}, top_k={v.config.top_k}, {v.config.embedder_name},{" "}
                      {v.config.generator_name}
                    </td>
                    <td className="py-2 pr-4 text-slate-700">{v.metrics.recall?.toFixed(3) ?? "—"}</td>
                    <td className="py-2 pr-4 text-slate-700">{v.metrics.faithfulness?.toFixed(3) ?? "—"}</td>
                    <td className="py-2 pr-4 text-xs text-slate-400">{v.timestamp.slice(0, 19).replace("T", " ")}</td>
                    <td className="py-2 pr-4">
                      <div className="flex gap-3 text-xs">
                        {v.name !== baseline && (
                          <button
                            onClick={() => handleSetBaseline(v.name)}
                            className="font-medium text-brand-600 hover:text-brand-700"
                          >
                            Set baseline
                          </button>
                        )}
                        <button onClick={() => handleDelete(v.name)} className="font-medium text-red-500 hover:text-red-700">
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {versions.length >= 2 && (
        <Card title="Check for regressions">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <LabeledSelect
              label="Version to check"
              value={checkName}
              onChange={setCheckName}
              options={versions.map((v) => ({ value: v.name, label: v.name }))}
            />
            <LabeledSelect
              label={`Against (default: baseline "${baseline}")`}
              value={againstName}
              onChange={setAgainstName}
              options={[
                { value: "", label: `Baseline (${baseline})` },
                ...versions.filter((v) => v.name !== checkName).map((v) => ({ value: v.name, label: v.name })),
              ]}
            />
          </div>
          <div className="mt-4">
            <Button onClick={handleCheck} disabled={checking || !checkName}>
              Check
            </Button>
          </div>
          {checking && (
            <div className="mt-4">
              <Spinner />
            </div>
          )}

          {comparison && (
            <div className="mt-6">
              <div className="mb-4">
                <span
                  className={`inline-block rounded-full px-3 py-1 text-sm font-medium ${
                    comparison.has_regression ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
                  }`}
                >
                  {comparison.has_regression ? "⚠️ Regression detected" : "✅ No regressions detected"}
                </span>
                <span className="ml-2 text-sm text-slate-500">
                  {comparison.new_version} vs {comparison.old_version}
                </span>
              </div>

              {comparison.regressions.length > 0 && (
                <MetricDeltaTable title="Regressions" entries={comparison.regressions} tone="red" />
              )}
              {comparison.improvements.length > 0 && (
                <MetricDeltaTable title="Improvements" entries={comparison.improvements} tone="green" />
              )}
              {comparison.unchanged.length > 0 && (
                <MetricDeltaTable title="Unchanged (within threshold)" entries={comparison.unchanged} tone="slate" />
              )}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

function MetricDeltaTable({ title, entries, tone }) {
  const toneClasses = {
    red: "text-red-600",
    green: "text-green-600",
    slate: "text-slate-500",
  };
  return (
    <div className="mb-4">
      <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">{title}</h3>
      <table className="w-full text-left text-sm">
        <tbody>
          {entries.map((e) => (
            <tr key={e.metric} className="border-b border-slate-100">
              <td className="py-1.5 pr-4 font-medium text-slate-700">{e.metric}</td>
              <td className="py-1.5 pr-4 text-slate-500">
                {e.old.toFixed(3)} → {e.new.toFixed(3)}
              </td>
              <td className={`py-1.5 pr-4 font-medium ${toneClasses[tone]}`}>
                {e.delta >= 0 ? "+" : ""}
                {e.delta.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
