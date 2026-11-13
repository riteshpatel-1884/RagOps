"use client";
import { useMemo, useState } from "react";
import { DATASET, VERSIONS, SECTIONS, NOTES } from "@/lib/experiments";
import { BarChart, LineChart, Scatter, Legend, COLORS } from "@/components/Charts";
import { compare, bestIndex, fmtVal, fmtDelta } from "@/lib/compare";

const HEADLINE = ["Answer correctness", "Hit rate @3", "False refusal rate", "Average cost / query"];
const multi = VERSIONS.length > 1;
// y-axis top for charts of small rates, so the bars stay readable (ticks stay clean quarters)
const niceMax = (groups) => [0.2, 0.4, 0.8, 1].find((x) => x >= Math.max(...groups.flatMap((g) => g.values)) * 1.15) || 1;

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

  const val = (label) => allMetrics.find((m) => m.label === label).values.map((x) => x.v);
  const vShort = VERSIONS.map((v) => ({ name: v.id, color: COLORS[v.id] || "#4f46e5" }));
  const typeLabels = VERSIONS[0].byType.map((t) => t.label);
  const byType = (key) => typeLabels.map((label, gi) => ({ label, values: VERSIONS.map((v) => v.byType[gi][key]) }));

  const summary = useMemo(() => {
    if (!multi) return [];
    return VERSIONS.slice(1).map((_, k) => {
      const i = k + 1;
      const from = mode === "baseline" ? 0 : i - 1;
      const rows = allMetrics.filter((m) => !m.noCompare).map((m) => ({ m, r: compare(m, from, i) }));
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
        <h1>{multi ? "RAG setups, measured" : "RAG baseline, measured"}</h1>
        <p>
          {DATASET.total} questions ({DATASET.answerable} answerable, {DATASET.unanswerable} unanswerable) over one
          document, the {DATASET.pages}-page Asterion employee handbook. Each version is scored on quality, retrieval,
          reliability, latency and cost.
        </p>
      </div>

      <div className="grid4">
        {headline.map((m) => {
          const v = m.values[last];
          return (
            <div key={m.label} className="card kpi">
              <div className="kpi-label">{m.label}</div>
              <div className="kpi-value">{fmtVal(v.v, m.fmt)}</div>
              <div className="kpi-sub">
                {multi ? (
                  <>
                    <span>{VERSIONS[last].id} vs {VERSIONS[0].id}</span>
                    <Delta metric={m} from={0} to={last} />
                  </>
                ) : v.lo != null ? (
                  <span>95% CI {v.lo.toFixed(2)}–{v.hi.toFixed(2)}</span>
                ) : (
                  <span>{m.sub || ""}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <h2 className="section-title">Charts</h2>
      <div className="grid2">
        <div className="card chart-card">
          <h3>Quality and retrieval</h3>
          <div className="sub">Higher is better</div>
          {multi && <Legend items={vShort} />}
          <BarChart
            series={vShort}
            groups={["Answer correctness", "Hit rate @3", "Context precision", "Context recall", "Context relevancy", "Faithfulness"].map((l) => ({
              label: l,
              values: val(l),
            }))}
          />
          <div className="takeaway">Context relevancy is the weak spot: chunks are about a page long, so most of what is sent is unrelated.</div>
        </div>
        <div className="card chart-card">
          <h3>Failure rates</h3>
          <div className="sub">Lower is better</div>
          {multi && <Legend items={vShort} />}
          <BarChart
            series={vShort}
            yMax={niceMax(["False refusal rate", "Hallucination rate"].map((l) => ({ values: val(l) })))}
            groups={["False refusal rate", "Hallucination rate"].map((l) => ({ label: l, values: val(l) }))}
          />
          <div className="takeaway">No hallucinations: every unanswerable question was correctly declined.</div>
        </div>
        <div className="card chart-card">
          <h3>Correctness by question type</h3>
          <div className="sub">Higher is better</div>
          {multi && <Legend items={vShort} />}
          <BarChart series={vShort} groups={byType("corr")} />
          <div className="takeaway">Multi-hop questions, which need facts from two places, are the weakest.</div>
        </div>
        <div className="card chart-card">
          <h3>False refusal by question type</h3>
          <div className="sub">Lower is better. Answerable questions the model wrongly refused</div>
          {multi && <Legend items={vShort} />}
          <BarChart series={vShort} yMax={niceMax(byType("fr"))} groups={byType("fr")} />
        </div>
        {multi && (
          <>
            <div className="card chart-card">
              <h3>Progress across versions</h3>
              <div className="sub">How the key metrics move</div>
              <Legend
                items={[
                  { name: "Answer correctness", color: "#4f46e5" },
                  { name: "Hit rate @3", color: "#0ea5e9" },
                  { name: "False refusal rate", color: "#ef4444" },
                ]}
              />
              <LineChart
                xLabels={VERSIONS.map((v) => v.id)}
                series={[
                  { name: "Answer correctness", color: "#4f46e5", values: val("Answer correctness"), below: true },
                  { name: "Hit rate @3", color: "#0ea5e9", values: val("Hit rate @3") },
                  { name: "False refusal rate", color: "#ef4444", values: val("False refusal rate"), dash: "6 5", below: true },
                ]}
              />
            </div>
            <div className="card chart-card">
              <h3>Cost vs correctness</h3>
              <div className="sub">Top left is best: cheaper and more correct</div>
              <Legend items={vShort} />
              <Scatter
                xMax={Math.max(...val("Average cost / query")) * 1.4}
                xFmt={(v) => `$${v.toFixed(5)}`}
                xTitle="Average cost per query"
                yTitle="Answer correctness"
                points={VERSIONS.map((v, i) => ({
                  label: v.id,
                  color: COLORS[v.id] || "#4f46e5",
                  x: val("Average cost / query")[i],
                  y: val("Answer correctness")[i],
                }))}
              />
            </div>
          </>
        )}
      </div>

      <h2 className="section-title">Pipeline used</h2>
      <div className="grid-auto">
        {VERSIONS.map((v) => (
          <div key={v.id} className="card version-card">
            <div className="vhead">
              <span className="vid">{v.id}</span>
              <span className="vtag">{v.tag}</span>
            </div>
            <dl>
              {v.pipeline.map(([k, val]) => (
                <div key={k}><dt>{k}</dt><dd>{val}</dd></div>
              ))}
              <div><dt>What changed</dt><dd>{v.change}</dd></div>
              <div><dt>Effect</dt><dd>{v.effect}</dd></div>
            </dl>
            <div className="code-title">Config and commands to reproduce</div>
            <pre className="code">{v.config}</pre>
          </div>
        ))}
      </div>

      <h2 className="section-title">By question type</h2>
      {VERSIONS.map((v) => (
        <section key={v.id} className="card table-card">
          <div className="card-head">
            <h2>{v.id} · {v.tag}</h2>
            <span className="muted">Answerable questions only. Unanswerable: 20 of 20 correctly declined</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Question type</th><th>Questions</th><th>Hit rate @3</th><th>Correctness</th><th>False refusal</th></tr>
              </thead>
              <tbody>
                {v.byType.map((t) => (
                  <tr key={t.label}>
                    <td className="metric-name">{t.label}</td>
                    <td>{t.n}</td>
                    <td>{t.hit.toFixed(2)}</td>
                    <td>{t.corr.toFixed(2)}</td>
                    <td>{t.fr.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}

      {multi && (
        <>
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
        </>
      )}

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
                      {m.values.map((v, i) => (
                        <td key={i} className={best === i ? "best" : ""}>
                          <div className="val">{fmtVal(v.v, m.fmt)}</div>
                          {v.lo != null && (
                            <div className="ci">
                              95% CI {v.lo.toFixed(2)}–{v.hi.toFixed(2)}
                            </div>
                          )}
                          {i > 0 && !m.noCompare && <Delta metric={m} from={fromOf(i)} to={i} />}
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
          {multi && <li>An asterisk (*) next to a change means the 95% confidence intervals overlap, so the difference may be noise.</li>}
          {multi && <li>The highlighted cell marks the best version for that metric.</li>}
        </ul>
      </div>
    </>
  );
}