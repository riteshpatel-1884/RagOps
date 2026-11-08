"use client";
import { useMemo, useState } from "react";
import { VERSIONS, SECTIONS, NOTES } from "@/lib/experiments";
import { compare, bestIndex, fmtVal, fmtDelta } from "@/lib/compare";

const HEADLINE = ["Answer correctness", "Hit rate @3", "False refusal rate", "Average cost / query"];

function Delta({ metric, from, to }) {
  const r = compare(metric, from, to);
  if (r.status === "same") return <span className="delta same">no change</span>;
  const arrow = r.diff > 0 ? "▲" : "▼";
  return (
    <span
      className={`delta ${r.status} ${r.significant ? "" : "weak"}`}
      title={r.significant ? "" : "Confidence intervals overlap, so this may be noise"}
    >
      {arrow} {fmtDelta(r.diff, metric.fmt)}
      {r.pct != null && metric.fmt !== "score" && ` (${r.pct > 0 ? "+" : ""}${r.pct.toFixed(0)}%)`}
      {!r.significant && "*"}
    </span>
  );
}

export default function EvalDashboard() {
  const [mode, setMode] = useState("baseline"); // "baseline" | "previous"

  const fromOf = (i) => (mode === "baseline" ? 0 : i - 1);
  const last = VERSIONS.length - 1;
  const allMetrics = useMemo(
    () => SECTIONS.flatMap((s) => s.metrics.map((m) => ({ ...m, section: s.title }))),
    []
  );
  const headline = useMemo(
    () => HEADLINE.map((label) => allMetrics.find((m) => m.label === label)).filter(Boolean),
    [allMetrics]
  );

  const summary = useMemo(() => {
    return [1, 2].map((i) => {
      const from = mode === "baseline" ? 0 : i - 1;
      const rows = allMetrics.map((m) => ({ m, r: compare(m, from, i) }));
      const size = (x) => Math.abs(x.r.pct ?? x.r.diff);
      return {
        idx: i,
        from,
        improved: rows.filter((x) => x.r.status === "improved").sort((a, b) => size(b) - size(a)),
        worse: rows.filter((x) => x.r.status === "worse").sort((a, b) => size(b) - size(a)),
        same: rows.filter((x) => x.r.status === "same").length,
      };
    });
  }, [mode, allMetrics]);

  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Evaluation</div>
        <h1>Three RAG setups, measured</h1>
        <p>
          The same 60 questions (48 answerable, 12 unanswerable) over a resume and a company handbook, run
          through each version and scored on the nine metrics below.
        </p>
      </div>

      <div className="grid4">
        {headline.map((m) => (
          <div key={m.label} className="card kpi">
            <div className="kpi-label">{m.label}</div>
            <div className="kpi-value">{fmtVal(m.values[last].v, m.fmt)}</div>
            <div className="kpi-sub">
              <span>{VERSIONS[last].id} vs {VERSIONS[0].id}</span>
              <Delta metric={m} from={0} to={last} />
            </div>
          </div>
        ))}
      </div>

      <h2 className="section-title">Versions</h2>
      <div className="grid3">
        {VERSIONS.map((v) => (
          <div key={v.id} className="card version-card">
            <div className="vhead">
              <span className="vid">{v.id}</span>
              <span className="vtag">{v.tag}</span>
            </div>
            <dl>
              <div><dt>Retriever</dt><dd>{v.retriever}</dd></div>
              <div><dt>Chunking</dt><dd>{v.chunking}</dd></div>
              <div><dt>Questions</dt><dd>{v.questions}</dd></div>
              <div><dt>What changed</dt><dd>{v.change}</dd></div>
              <div><dt>Effect</dt><dd>{v.effect}</dd></div>
            </dl>
          </div>
        ))}
      </div>

      <h2 className="section-title">What improved and what got worse</h2>
      <div className="toolbar">
        <span className="muted">Compare each version against</span>
        <div className="seg">
          <button className={mode === "baseline" ? "seg-btn active" : "seg-btn"} onClick={() => setMode("baseline")}>
            Baseline (v1)
          </button>
          <button className={mode === "previous" ? "seg-btn active" : "seg-btn"} onClick={() => setMode("previous")}>
            Previous version
          </button>
        </div>
      </div>

      <div className="grid2">
        {summary.map((s) => (
          <div key={s.idx} className="card">
            <div className="card-head">
              <h2>
                {VERSIONS[s.idx].id} <span className="muted">vs {VERSIONS[s.from].id}</span>
              </h2>
              <div className="counts">
                <span className="pill good">{s.improved.length} improved</span>
                <span className="pill bad">{s.worse.length} worse</span>
                <span className="pill neutral">{s.same} same</span>
              </div>
            </div>

            <div className="cols">
              <div>
                <h3 className="good-t">Improved</h3>
                <ul className="list">
                  {s.improved.length === 0 && <li className="muted">Nothing improved</li>}
                  {s.improved.map(({ m, r }) => (
                    <li key={m.section + m.label}>
                      <span>{m.label}</span>
                      <span className={`delta improved ${r.significant ? "" : "weak"}`}>
                        {fmtDelta(r.diff, m.fmt)}{!r.significant && "*"}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="bad-t">Got worse</h3>
                <ul className="list">
                  {s.worse.length === 0 && <li className="muted">Nothing got worse</li>}
                  {s.worse.map(({ m, r }) => (
                    <li key={m.section + m.label}>
                      <span>{m.label}</span>
                      <span className={`delta worse ${r.significant ? "" : "weak"}`}>
                        {fmtDelta(r.diff, m.fmt)}{!r.significant && "*"}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h2 className="section-title">Metrics</h2>
      {SECTIONS.map((sec) => (
        <section key={sec.id} className="card table-card">
          <div className="card-head">
            <h2>{sec.title}</h2>
            <span className="muted">{sec.note}</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  {VERSIONS.map((v) => (
                    <th key={v.id}>{v.id}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sec.metrics.map((m) => {
                  const best = bestIndex(m);
                  return (
                    <tr key={m.label}>
                      <td className="metric-name">
                        {m.label}
                        <span className="hint">{m.better === "up" ? "Higher is better" : "Lower is better"}</span>
                      </td>
                      {m.values.map((val, i) => (
                        <td key={i} className={best === i ? "best" : ""}>
                          <div className="val">{fmtVal(val.v, m.fmt)}</div>
                          {val.lo != null && (
                            <div className="ci">
                              95% CI {val.lo.toFixed(2)}–{val.hi.toFixed(2)}
                            </div>
                          )}
                          {i > 0 && <Delta metric={m} from={fromOf(i)} to={i} />}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ))}

      <div className="card notes">
        <h2>Notes and caveats</h2>
        <ul>
          {NOTES.map((n) => (
            <li key={n}>{n}</li>
          ))}
          <li>An asterisk (*) next to a change means the 95% confidence intervals overlap, so the difference may be noise.</li>
          <li>The highlighted cell marks the best version for that metric.</li>
        </ul>
      </div>
    </>
  );
}