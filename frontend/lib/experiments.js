// c(mean, lo, hi) = value with 95% bootstrap CI, p(v) = plain value
const c = (v, lo, hi) => ({ v, lo, hi });
const p = (v) => ({ v });

// Everything here is a record of what was run and what was measured. To add a version, append one entry to
// VERSIONS and one value to every metric in SECTIONS (same order).
export const DATASET = {
  pdf: "Asterion_Employee_Handbook.pdf",
  pages: 27,
  total: 105,
  answerable: 85,
  unanswerable: 20,
  judged: 43,
};

export const VERSIONS = [
  {
    id: "v1",
    tag: "Baseline",
    change: "Starting point: static character splitting, page-sized chunks, plain vector search.",
    effect:
      "Finds the right page and never invents answers, but every chunk is a page of mostly unrelated text (context relevancy 0.11), and questions that need facts from two places fail most often (multi-hop false refusal 20%).",
    // [label, value] rows shown on the version card, in pipeline order
    pipeline: [
      ["Corpus", "Asterion_Employee_Handbook.pdf (27 pages, 502 text blocks after removing repeated headers, footers and page numbers)"],
      ["Chunking", "Static character splitting (LangChain RecursiveCharacterTextSplitter): 2,000 characters (about 500 tokens), 200 overlap"],
      ["Chunks", "41 chunks, median 1,801 characters, max 1,996 (roughly one page each)"],
      ["Embedding", "gemini-embedding-001, 768 dimensions, L2-normalised, cached locally"],
      ["Vector store", "Qdrant Cloud, cosine distance, one collection per version (asterion_v1_cfef5bded2, 41 points)"],
      ["Retrieval", "Dense vector search, top 3 chunks. No reranker, no keyword search"],
      ["Generator", "Groq openai/gpt-oss-120b, temperature 0, streamed. Answers only from the chunks, cites [p. N], says \"I could not find this in the handbook.\" when the answer is missing"],
      ["Judge", "DeepEval with Groq openai/gpt-oss-120b as judge (6 metrics), correctness via GEval with threshold 0.7"],
      ["Questions", "105 (40 single fact, 25 multi-part, 20 multi-hop, 20 unanswerable). Judge metrics on 43 of the 85 answerable (every 2nd); other metrics on all 105"],
    ],
    config: `"v1": {
  "chunking":  {"strategy": "static", "chunk_chars": 2000, "overlap_chars": 200},
  "embedding": {"model": "gemini-embedding-001", "dim": 768},
  "retrieval": {"mode": "vector", "top_k": 3}
}

python -m app.versions.cli run v1
python -m app.versions.cli judge v1 --every 2 --sleep 10
python -m app.versions.cli summary v1
python -m app.versions.cli save v1`,
    byType: [
      { label: "Single fact", n: 40, hit: 0.95, fr: 0.0, corr: 0.94 },
      { label: "Multi-part", n: 25, hit: 0.96, fr: 0.04, corr: 0.92 },
      { label: "Multi-hop", n: 20, hit: 0.95, fr: 0.2, corr: 0.75 },
    ],
  },
  {
    id: "v2",
    tag: "Smaller chunks",
    change: "Only two settings changed: chunk size 2,000 to 1,000 characters, and top 3 to top 4 chunks. Embedding, vector search, generator, judge and questions are identical to v1.",
    effect:
      "Each chunk carries less unrelated text and one more chunk is sent, so questions that need two facts are answered more often: false refusal fell from 5.9% to 2.4% and multi-hop false refusal from 20% to 10%. Context recall reached 1.00 and cost fell 12%. Most differences are inside the confidence intervals, so read them as a consistent trend, not proof.",
    pipeline: [
      ["Corpus", "Asterion_Employee_Handbook.pdf (same as v1)"],
      ["Chunking", "Static character splitting: 1,000 characters (about 250 tokens), 150 overlap"],
      ["Chunks", "70 chunks, median 817 characters, max 992"],
      ["Embedding", "gemini-embedding-001, 768 dimensions (same as v1)"],
      ["Vector store", "Qdrant Cloud, cosine distance (collection asterion_v2_ead2fd666b, 70 points)"],
      ["Retrieval", "Dense vector search, top 4 chunks. No reranker, no keyword search"],
      ["Generator", "Groq openai/gpt-oss-120b, temperature 0 (same as v1)"],
      ["Judge", "DeepEval with Groq openai/gpt-oss-120b (same as v1)"],
      ["Questions", "Same 105 questions. Judge metrics on the same 43 answerable questions as v1"],
    ],
    config: `"v2": {
  "chunking":  {"strategy": "static", "chunk_chars": 1000, "overlap_chars": 150},
  "embedding": {"model": "gemini-embedding-001", "dim": 768},
  "retrieval": {"mode": "vector", "top_k": 4}
}

python -m app.versions.cli run v2
python -m app.versions.cli judge v2 --every 2 --sleep 10
python -m app.versions.cli summary v2
python -m app.versions.cli save v2`,
    byType: [
      { label: "Single fact", n: 40, hit: 0.95, fr: 0.0, corr: 0.9 },
      { label: "Multi-part", n: 25, hit: 0.96, fr: 0.0, corr: 1.0 },
      { label: "Multi-hop", n: 20, hit: 0.95, fr: 0.1, corr: 0.84 },
    ],
  },
];

// better: "up" = higher is better, "down" = lower is better
// fmt: score | cost | ms
export const SECTIONS = [
  {
    id: "quality",
    title: "Answer and context quality",
    note: "DeepEval judge on 43 answerable questions. Mean with 95% CI",
    metrics: [
      { label: "Answer correctness", better: "up", fmt: "score", values: [c(0.893, 0.802, 0.967), c(0.914, 0.847, 0.965)] },
      { label: "Faithfulness", better: "up", fmt: "score", values: [c(1.0, 1.0, 1.0), c(0.996, 0.988, 1.0)] },
      { label: "Answer relevancy", better: "up", fmt: "score", values: [c(0.989, 0.969, 1.0), c(0.993, 0.982, 1.0)] },
      { label: "Context precision", better: "up", fmt: "score", values: [c(0.915, 0.849, 0.973), c(0.906, 0.839, 0.961)] },
      { label: "Context recall", better: "up", fmt: "score", values: [c(0.977, 0.93, 1.0), c(1.0, 1.0, 1.0)] },
      { label: "Context relevancy", better: "up", fmt: "score", values: [c(0.107, 0.075, 0.146), c(0.146, 0.107, 0.192)] },
    ],
  },
  {
    id: "retrieval",
    title: "Retrieval and reliability",
    note: "Judge-free, all 105 questions. Hit rate and false refusal use the 85 answerable, hallucination and abstention the 20 unanswerable",
    metrics: [
      { label: "Hit rate @3", better: "up", fmt: "score", values: [c(0.953, 0.906, 0.988), c(0.953, 0.906, 0.988)] },
      { label: "False refusal rate", better: "down", fmt: "score", values: [c(0.059, 0.012, 0.118), c(0.024, 0.0, 0.059)] },
      { label: "Hallucination rate", better: "down", fmt: "score", values: [c(0.0, 0.0, 0.0), c(0.0, 0.0, 0.0)] },
      { label: "Abstention accuracy", better: "up", fmt: "score", values: [c(1.0, 1.0, 1.0), c(1.0, 1.0, 1.0)] },
    ],
  },
  {
    id: "latency",
    title: "Latency",
    note: "v1 and v2 were measured on different days; Groq and Gemini load changes a lot, so do not read the latency gap as an effect of the setup",
    metrics: [
      { label: "Retrieval time (p50)", better: "down", fmt: "ms", noCompare: true, values: [p(2724), p(651)] },
      { label: "Time to first token (p50)", better: "down", fmt: "ms", noCompare: true, values: [p(7166), p(3469)] },
      { label: "Time to first token (p95)", better: "down", fmt: "ms", noCompare: true, values: [p(10622), p(6774)] },
      { label: "Total time (p50)", better: "down", fmt: "ms", noCompare: true, values: [p(7398), p(3538)] },
      { label: "Total time (p95)", better: "down", fmt: "ms", noCompare: true, values: [p(10726), p(7261)] },
    ],
  },
  {
    id: "cost",
    title: "Cost",
    note: "Groq gpt-oss-120b list price ($0.15 per 1M input, $0.60 per 1M output tokens). Lower is better",
    metrics: [
      { label: "Average cost / query", better: "down", fmt: "cost", sub: "$0.25 per 1,000 queries", values: [p(0.000249), p(0.000220)] },
    ],
  },
];

export const NOTES = [
  "105 questions over one document (the 27-page Asterion employee handbook): 40 single fact, 25 multi-part, 20 multi-hop, 20 unanswerable. Every expected answer has verbatim evidence from a specific page. CI = 95% bootstrap interval.",
  "Judge metrics (correctness, faithfulness, relevancy, context precision, recall and relevancy) were run on the same 43 of the 85 answerable questions (every second one) for every version, to stay within Groq's free limits. With n=43 the intervals are about plus or minus 0.1, so most differences between v1 and v2 are inside the noise.",
  "v2 changed only chunk size (2,000 to 1,000 characters) and top_k (3 to 4). It improved false refusal (5.9% to 2.4%), multi-hop false refusal (20% to 10%), multi-hop correctness (0.75 to 0.84), context recall (0.98 to 1.00), context relevancy (0.11 to 0.15) and cost (-12%). Single-fact correctness slipped from 0.94 to 0.90, which is about two questions.",
  "Context relevancy stays low for both versions (0.11 and 0.15). The metric counts the share of retrieved statements that are relevant, and these answers sit in one sentence, so only sentence-sized contexts would score high, at the price of losing multi-fact questions.",
  "Hit rate @3 looks only at the first 3 retrieved chunks. v2 retrieves 4, so its hit rate @3 (0.953) understates it; an offline retrieval test counting all 4 gave 0.988.",
  "A FlashRank reranking step was tested offline (no LLM) and not kept: with 1,000-character chunks it lowered the share of multi-evidence questions where every needed fact was retrieved (0.775 vs 0.875 for plain vector search).",
  "Faithfulness is about 1.0 because the model stays inside the retrieved text. It cannot tell you the retrieval was right; correctness and hit rate do.",
  "Hallucination is 0% and abstention accuracy 100% for both versions: all 20 unanswerable questions were answered with \"I could not find this in the handbook.\"",
  "Remaining weak spot: multi-hop questions that need rows from different parts of the employee directory (for example a person and their manager). Those facts sit in different chunks, and neither setup reliably retrieves both.",
];