// c(mean, lo, hi) = value with 95% bootstrap CI, p(v) = plain value
const c = (v, lo, hi) => ({ v, lo, hi });
const p = (v) => ({ v });

export const VERSIONS = [
  { id: "v1", tag: "Baseline", retriever: "k=3", chunking: "raw PDF blocks", questions: 41 },
  { id: "v2", tag: "Smaller retriever", retriever: "k=1", chunking: "raw PDF blocks", questions: 41 },
  { id: "v3", tag: "Merged chunks", retriever: "k=3", chunking: "merged blocks (700 chars)", questions: 41 },
];

// better: "up" = higher is better, "down" = lower is better
// fmt: score | ms | int | usd
export const SECTIONS = [
  {
    id: "judge",
    title: "Judge metrics (answerable questions)",
    note: "Mean with 95% CI",
    metrics: [
      { label: "Answer relevancy", better: "up", fmt: "score", values: [c(0.988, 0.97, 1.0), c(1.0, 1.0, 1.0), c(0.988, 0.96, 1.0)] },
      { label: "Faithfulness", better: "up", fmt: "score", values: [c(0.951, 0.88, 1.0), c(0.927, 0.83, 1.0), c(0.976, 0.93, 1.0)] },
      { label: "Context relevancy", better: "up", fmt: "score", values: [c(0.289, 0.24, 0.34), c(0.341, 0.21, 0.49), c(0.194, 0.15, 0.25)] },
      { label: "Correctness", better: "up", fmt: "score", values: [c(0.551, 0.43, 0.66), c(0.185, 0.1, 0.29), c(0.807, 0.73, 0.87)] },
      { label: "Completeness", better: "up", fmt: "score", values: [c(0.602, 0.48, 0.71), c(0.224, 0.12, 0.35), c(0.859, 0.78, 0.92)] },
    ],
  },
  {
    id: "retrieval",
    title: "Retrieval (at each version's own k)",
    note: "Mean with 95% CI",
    metrics: [
      { label: "Hit rate", better: "up", fmt: "score", values: [c(0.829, 0.71, 0.93), c(0.463, 0.32, 0.63), c(1.0, 1.0, 1.0)] },
      { label: "Recall", better: "up", fmt: "score", values: [c(0.709, 0.58, 0.83), c(0.356, 0.23, 0.5), c(0.951, 0.9, 0.99)] },
      { label: "Precision", better: "up", fmt: "score", values: [c(0.301, 0.25, 0.35), c(0.463, 0.32, 0.63), c(0.439, 0.39, 0.5)] },
      { label: "MRR", better: "up", fmt: "score", values: [c(0.638, 0.52, 0.76), c(0.463, 0.32, 0.63), c(0.846, 0.77, 0.92)] },
    ],
  },
  {
    id: "curve",
    title: "Retrieval curve at fixed depths",
    note: "Hit rate and recall at the same k for every version",
    metrics: [
      { label: "Hit @ k=1", better: "up", fmt: "score", values: [p(0.46), p(0.46), p(0.71)] },
      { label: "Recall @ k=1", better: "up", fmt: "score", values: [p(0.36), p(0.36), p(0.64)] },
      { label: "Hit @ k=3", better: "up", fmt: "score", values: [p(0.83), p(0.83), p(1.0)] },
      { label: "Recall @ k=3", better: "up", fmt: "score", values: [p(0.71), p(0.71), p(0.95)] },
      { label: "Hit @ k=5", better: "up", fmt: "score", values: [p(0.93), p(0.93), p(1.0)] },
      { label: "Recall @ k=5", better: "up", fmt: "score", values: [p(0.89), p(0.89), p(0.98)] },
    ],
  },
  {
    id: "generation",
    title: "Generation (judge-free)",
    note: "Mean with 95% CI",
    metrics: [
      { label: "Key-fact coverage", better: "up", fmt: "score", values: [c(0.655, 0.52, 0.79), c(0.297, 0.17, 0.44), c(0.862, 0.77, 0.94)] },
      { label: "Abstention accuracy", better: "up", fmt: "score", values: [c(0.9, 0.7, 1.0), c(1.0, 1.0, 1.0), c(0.9, 0.7, 1.0)] },
      { label: "Hallucination rate", better: "down", fmt: "score", values: [c(0.1, 0.0, 0.3), c(0.0, 0.0, 0.0), c(0.1, 0.0, 0.3)] },
      { label: "False refusal", better: "down", fmt: "score", values: [c(0.171, 0.07, 0.29), c(0.659, 0.51, 0.8), c(0.0, 0.0, 0.0)] },
      { label: "Context grounding", better: "up", fmt: "score", values: [c(0.998, 0.99, 1.0), c(1.0, 1.0, 1.0), c(0.976, 0.93, 1.0)] },
      { label: "Paraphrase consistency", better: "up", fmt: "score", values: [p(0.533), p(0.8), p(0.7)] },
    ],
  },
  {
    id: "performance",
    title: "Performance & cost",
    note: "Lower is better",
    metrics: [
      { label: "Retrieval latency p50", better: "down", fmt: "ms", values: [p(192), p(199), p(219)] },
      { label: "Generation latency p50", better: "down", fmt: "ms", values: [p(1218), p(1317), p(1487)] },
      { label: "Total latency p50", better: "down", fmt: "ms", values: [p(1514), p(1580), p(1886)] },
      { label: "Total latency p95", better: "down", fmt: "ms", values: [p(2933), p(10462), p(6496)] },
      { label: "Input tokens / query", better: "down", fmt: "int", values: [p(392), p(331), p(687)] },
      { label: "Output tokens / query", better: "down", fmt: "int", values: [p(288), p(224), p(310)] },
      { label: "Cost / 1,000 queries", better: "down", fmt: "usd", values: [p(0.232), p(0.184), p(0.289)] },
    ],
  },
  {
    id: "category",
    title: "Correctness by question category",
    note: "Where each version wins or loses",
    metrics: [
      { label: "multi_hop", better: "up", fmt: "score", values: [p(0.067), p(0.067), p(0.517)] },
      { label: "multi_part", better: "up", fmt: "score", values: [p(0.545), p(0.105), p(0.815)] },
      { label: "single_fact", better: "up", fmt: "score", values: [p(0.753), p(0.34), p(0.913)] },
    ],
  },
];

export const NOTES = [
  "CI = 95% bootstrap interval. Overlapping intervals mean the difference may be noise.",
  "v2 (k=1) answers fewer questions: false refusal jumps to 0.659 and correctness collapses.",
  "v3 (merged 700-char blocks) is the best on correctness, completeness, hit rate and multi-hop questions, but costs more tokens and latency and has lower context relevancy.",
];
