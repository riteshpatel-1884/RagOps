export function fmtVal(v, fmt) {
  if (v == null) return "–";
  switch (fmt) {
    case "ms":
      return `${Math.round(v).toLocaleString()} ms`;
    case "int":
      return Math.round(v).toLocaleString();
    case "usd":
      return `$${v.toFixed(3)}`;
    default:
      return v.toFixed(3);
  }
}

export function fmtDelta(d, fmt) {
  const sign = d > 0 ? "+" : d < 0 ? "−" : "";
  const a = Math.abs(d);
  switch (fmt) {
    case "ms":
      return `${sign}${Math.round(a).toLocaleString()} ms`;
    case "int":
      return `${sign}${Math.round(a).toLocaleString()}`;
    case "usd":
      return `${sign}$${a.toFixed(3)}`;
    default:
      return `${sign}${a.toFixed(3)}`;
  }
}

// Compare metric.values[fromIdx] -> metric.values[toIdx]
export function compare(metric, fromIdx, toIdx) {
  const a = metric.values[fromIdx];
  const b = metric.values[toIdx];
  const diff = b.v - a.v;

  if (Math.abs(diff) < 1e-9) return { diff: 0, pct: 0, status: "same", significant: false };

  const good = metric.better === "up" ? diff > 0 : diff < 0;
  const hasCI = a.lo != null && b.lo != null;
  const overlap = hasCI && a.lo <= b.hi && b.lo <= a.hi;

  return {
    diff,
    pct: a.v !== 0 ? (diff / a.v) * 100 : null,
    status: good ? "improved" : "worse",
    significant: !overlap, // CIs overlapping => may be noise
  };
}

// Index of the best version for a metric (null if all equal)
export function bestIndex(metric) {
  const vals = metric.values.map((x) => x.v);
  const target = metric.better === "up" ? Math.max(...vals) : Math.min(...vals);
  const winners = vals.filter((v) => v === target).length;
  if (winners === vals.length) return null;
  return vals.indexOf(target);
}
