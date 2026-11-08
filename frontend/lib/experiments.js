// c(mean, lo, hi) = value with 95% bootstrap CI, p(v) = plain value
const c = (v, lo, hi) => ({ v, lo, hi });
const p = (v) => ({ v });

// 60-question evaluation set (48 answerable + 12 unanswerable) over 2 PDFs: a resume and a 12-page company handbook.
export const VERSIONS = [
  {
    id: "v1",
    tag: "Baseline",
    retriever: "k=3",
    chunking: "raw PDF blocks",
    questions: "48 answerable + 12 unanswerable",
    change: "Starting point: raw PDF blocks as chunks, top 3 chunks sent to the LLM.",
    effect: "Chunks are too small to hold a full answer, so multi-part and multi-hop questions fail and the model often says 'not found'.",
  },
  {
    id: "v2",
    tag: "Smaller retriever",
    retriever: "k=1",
    chunking: "raw PDF blocks",
    questions: "48 answerable + 12 unanswerable",
    change: "Same as v1 but only the top 1 chunk is sent to the LLM.",
    effect: "Cheapest, but 60% of answerable questions are refused. Zero hallucination only because it refuses almost everything.",
  },
  {
    id: "v3",
    tag: "Merged chunks + cleaner search",
    retriever: "k=3",
    chunking: "merged blocks (700 chars)",
    questions: "48 answerable + 12 unanswerable",
    change: "Two changes: (1) neighbouring blocks merged into ~700-character chunks, (2) the company name is removed from the search query (the LLM still sees the full question).",
    effect: "Chunks now carry whole answers, and the topic chunk is no longer outranked by generic 'about the company' text. Best quality, at the price of more tokens.",
  },
];

// better: "up" = higher is better, "down" = lower is better
// fmt: score | cost
export const SECTIONS = [
  {
    id: "quality",
    title: "Answer quality",
    note: "Judged on the 48 answerable questions. Mean with 95% CI",
    metrics: [
      { label: "Answer correctness", better: "up", fmt: "score", values: [c(0.435, 0.315, 0.558), c(0.258, 0.163, 0.356), c(0.798, 0.719, 0.873)] },
      { label: "Faithfulness", better: "up", fmt: "score", values: [c(1.0, 1.0, 1.0), c(1.0, 1.0, 1.0), c(1.0, 1.0, 1.0)] },
      { label: "Context relevancy", better: "up", fmt: "score", values: [c(0.33, 0.247, 0.413), c(0.505, 0.359, 0.646), c(0.165, 0.125, 0.214)] },
    ],
  },
  {
    id: "retrieval",
    title: "Retrieval (top 3 chunks)",
    note: "Judge-free. Same depth (k=3) for every version",
    metrics: [
      { label: "Hit rate @3", better: "up", fmt: "score", values: [c(0.667, 0.521, 0.792), c(0.667, 0.521, 0.792), c(0.979, 0.938, 1.0)] },
      { label: "Recall @3", better: "up", fmt: "score", values: [c(0.528, 0.398, 0.656), c(0.528, 0.398, 0.656), c(0.918, 0.851, 0.972)] },
    ],
  },
  {
    id: "reliability",
    title: "Reliability",
    note: "Hallucination and abstention use the 12 unanswerable questions; false refusal uses the 48 answerable ones",
    metrics: [
      { label: "Hallucination rate", better: "down", fmt: "score", values: [c(0.167, 0.0, 0.417), c(0.0, 0.0, 0.0), c(0.083, 0.0, 0.25)] },
      { label: "Abstention accuracy", better: "up", fmt: "score", values: [c(0.833, 0.583, 1.0), c(1.0, 1.0, 1.0), c(0.917, 0.75, 1.0)] },
      { label: "False refusal rate", better: "down", fmt: "score", values: [c(0.396, 0.25, 0.542), c(0.604, 0.458, 0.75), c(0.042, 0.0, 0.104)] },
    ],
  },
  {
    id: "cost",
    title: "Cost",
    note: "Groq gpt-oss-120b list price. Lower is better",
    metrics: [
      { label: "Average cost / query", better: "down", fmt: "cost", values: [p(0.000222), p(0.000158), p(0.000303)] },
    ],
  },
];

export const NOTES = [
  "60 questions over 2 documents (a resume and a 12-page company handbook). CI = 95% bootstrap interval; overlapping intervals mean the difference may be noise.",
  "v2 hallucinates 0% only because it refuses 60% of answerable questions. Read hallucination together with false refusal.",
  "Hit rate and recall @3 are identical for v1 and v2 because they share the same chunks. v2 just sends only the top 1 chunk to the LLM (hit rate @1 is 0.48).",
  "Faithfulness is 1.0 for every version because answers stay grounded in whatever was retrieved, even when the retrieval was wrong. Correctness is the metric that shows quality.",
  "Context relevancy falls in v3 because it sends bigger chunks; the judge rewards small context. Treat it as a side effect, not a regression in answers.",
  "v3 bundles two changes (merged chunks + cleaner search query), so v3 vs v1 does not isolate either one. The query change was chosen after inspecting failures on these same 60 questions, so v3's gain is probably a little optimistic.",
  "v3's weak spot: cross-document questions (correctness 0.55). It also costs about 36% more per query than v1 because of the larger chunks.",
];