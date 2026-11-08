// Dependency-free SVG charts. All of them scale to the width of their card.
export const COLORS = { v1: "#94a3b8", v2: "#0ea5e9", v3: "#4f46e5" };
const GRID = "#e5e7eb";
const TEXT = "#64748b";
const W = 560, H = 300;
const M = { l: 44, r: 16, t: 14, b: 48 };
const PW = W - M.l - M.r;
const PH = H - M.t - M.b;

const r2 = (v) => (Math.round(v * 100) / 100).toFixed(2);

const wrap = (label, max = 14) => {
  const words = label.split(" ");
  const lines = [];
  let cur = "";
  for (const w of words) {
    if ((cur + " " + w).trim().length > max && cur) { lines.push(cur); cur = w; }
    else cur = (cur + " " + w).trim();
  }
  if (cur) lines.push(cur);
  return lines;
};

function Grid({ yMax, fmt }) {
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * yMax);
  return (
    <g>
      {ticks.map((t) => {
        const y = M.t + PH - (t / yMax) * PH;
        return (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y} y2={y} stroke={GRID} strokeWidth="1" />
            <text x={M.l - 8} y={y + 4} textAnchor="end" fontSize="11" fill={TEXT}>{fmt(t)}</text>
          </g>
        );
      })}
    </g>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.name} className="legend-item">
          <i style={{ background: it.color }} /> {it.name}
        </span>
      ))}
    </div>
  );
}

// groups: [{ label, values: [..] }]  series: [{ name, color }]
export function BarChart({ groups, series, yMax = 1, fmt = r2 }) {
  const gw = PW / groups.length;
  const bw = Math.min(34, (gw * 0.72) / series.length);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img">
      <Grid yMax={yMax} fmt={(v) => (yMax <= 1 ? v.toFixed(2).replace(/0$/, "") : v.toFixed(2))} />
      {groups.map((g, gi) => {
        const cx = M.l + gi * gw + gw / 2;
        const start = cx - (bw * series.length) / 2;
        return (
          <g key={g.label}>
            {g.values.map((v, si) => {
              const h = (v / yMax) * PH;
              const x = start + si * bw;
              const y = M.t + PH - h;
              return (
                <g key={si}>
                  <rect x={x + 1} y={y} width={bw - 2} height={Math.max(h, 0)} rx="3" fill={series[si].color}>
                    <title>{`${series[si].name} · ${g.label}: ${fmt(v)}`}</title>
                  </rect>
                  <text x={x + bw / 2} y={y - 5} textAnchor="middle" fontSize="10.5" fontWeight="600" fill="#334155">
                    {fmt(v)}
                  </text>
                </g>
              );
            })}
            {wrap(g.label).map((line, li) => (
              <text key={li} x={cx} y={M.t + PH + 18 + li * 13} textAnchor="middle" fontSize="11.5" fill="#334155">{line}</text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

// xLabels: ["v1","v2","v3"]  series: [{ name, color, values, dash? }]
export function LineChart({ xLabels, series, yMax = 1, fmt = r2 }) {
  const step = PW / (xLabels.length - 1 || 1);
  const pad = 28; // keep first / last point away from the axes
  const sx = (i) => M.l + pad + (i * (PW - pad * 2)) / (xLabels.length - 1 || 1);
  const sy = (v) => M.t + PH - (v / yMax) * PH;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img">
      <Grid yMax={yMax} fmt={(v) => v.toFixed(2).replace(/0$/, "")} />
      {xLabels.map((l, i) => (
        <text key={l} x={sx(i)} y={M.t + PH + 32} textAnchor="middle" fontSize="12" fill="#334155">{l}</text>
      ))}
      {series.map((s) => (
        <g key={s.name}>
          <polyline
            fill="none"
            stroke={s.color}
            strokeWidth="2.5"
            strokeLinejoin="round"
            strokeLinecap="round"
            strokeDasharray={s.dash || undefined}
            points={s.values.map((v, i) => `${sx(i)},${sy(v)}`).join(" ")}
          />
          {s.values.map((v, i) => (
            <g key={i}>
              <circle cx={sx(i)} cy={sy(v)} r="4.5" fill="#fff" stroke={s.color} strokeWidth="2.5">
                <title>{`${s.name} · ${xLabels[i]}: ${fmt(v)}`}</title>
              </circle>
              {s.labels !== false && (
                <text x={sx(i)} y={s.below ? sy(v) + 18 : sy(v) - 10} textAnchor="middle" fontSize="10.5" fontWeight="600" fill={s.color}>{fmt(v)}</text>
              )}
            </g>
          ))}
        </g>
      ))}
    </svg>
  );
}

// points: [{ label, x, y, color }]
export function Scatter({ points, xMax, yMax = 1, xFmt, yFmt = r2, xTitle, yTitle }) {
  const sx = (v) => M.l + 20 + (v / xMax) * (PW - 40);
  const sy = (v) => M.t + PH - (v / yMax) * PH;
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * xMax);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img">
      <Grid yMax={yMax} fmt={(v) => v.toFixed(2).replace(/0$/, "")} />
      {xTicks.map((t) => (
        <text key={t} x={sx(t)} y={M.t + PH + 18} textAnchor="middle" fontSize="11" fill={TEXT}>{xFmt(t)}</text>
      ))}
      <text x={M.l + PW / 2} y={H - 6} textAnchor="middle" fontSize="11.5" fill={TEXT}>{xTitle}</text>
      <text x={8} y={M.t + PH / 2} fontSize="11.5" fill={TEXT} transform={`rotate(-90 8 ${M.t + PH / 2})`} textAnchor="middle">{yTitle}</text>
      {points.map((p) => (
        <g key={p.label}>
          <circle cx={sx(p.x)} cy={sy(p.y)} r="9" fill={p.color} fillOpacity="0.18" stroke={p.color} strokeWidth="2.5">
            <title>{`${p.label}: ${xFmt(p.x)} per query, correctness ${yFmt(p.y)}`}</title>
          </circle>
          <text x={sx(p.x)} y={sy(p.y) - 16} textAnchor="middle" fontSize="12.5" fontWeight="700" fill={p.color}>{p.label}</text>
        </g>
      ))}
    </svg>
  );
}