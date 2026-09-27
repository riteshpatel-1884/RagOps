// Shared API client for the RAG Evaluation & Optimization Engine backend.
// Every page imports from here rather than calling fetch() directly, so
// the base URL and error handling only live in one place.

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiError extends Error {}

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, {
      headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
      ...options,
    });
  } catch (e) {
    throw new ApiError(
      `Could not reach the API at ${API_URL}. Is the backend running? (uvicorn api:app --port 8000)`
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore parse failure, fall back to statusText */
    }
    throw new ApiError(detail);
  }
  return res.json();
}

// --- Shape reference (for documentation only — no runtime effect) ----------
//
// ConfigOption:            { value, label }
// ConfigOptions:           { embedders, generators, judges } — arrays of ConfigOption
// PipelineConfigDict:      { chunk_size, chunk_overlap, embedder_name, top_k, collection_name?, generator_name }
// TestDatasetItem:         { id, question, relevant_doc_ids }
// RetrievalEvalResult:     { overall, per_query: [{ id, question, top_hit_doc, expected_docs, first_relevant_rank, ...metrics }] }
// GenerationEvalResult:    { avg_faithfulness, avg_relevance, rows: [{ id, question, answer, faithfulness, faithfulness_explanation, relevance, relevance_explanation }] }
// AskResult:               { question, answer, retrieved_hits: [{ chunk_id, doc_id, text, score }] }
// ExperimentRow:           { config, recall?, precision?, mrr?, ndcg?, elapsed_seconds?, faithfulness?, relevance?, error? }
// ExperimentResponse:      { results, failed, sort_by, filename }
// ExperimentListItem:      { filename, modified }
// CorpusStatus:            { source: "demo"|"custom", doc_count, test_dataset_count, has_seed_demo }

// --- Endpoints ---------------------------------------------------------------

export const api = {
  configOptions: () => request("/api/config-options"),

  uploadDocuments: (files) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request("/api/documents/upload", {
      method: "POST",
      headers: {}, // let the browser set the multipart boundary
      body: form,
    });
  },

  getDocuments: () => request("/api/documents"),

  getCorpusStatus: () => request("/api/corpus/status"),

  resetDemoCorpus: () => request("/api/corpus/reset-demo", { method: "POST" }),

  saveTestDataset: (items) =>
    request("/api/test-dataset", { method: "POST", body: JSON.stringify(items) }),

  uploadTestDatasetFile: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/test-dataset/upload", { method: "POST", headers: {}, body: form });
  },

  getTestDataset: () => request("/api/test-dataset"),

  evaluateRetrieval: (config) =>
    request("/api/evaluate/retrieval", { method: "POST", body: JSON.stringify(config) }),

  evaluateGeneration: (config) =>
    request("/api/evaluate/generation", { method: "POST", body: JSON.stringify(config) }),

  ask: (payload) =>
    request("/api/ask", { method: "POST", body: JSON.stringify(payload) }),

  runExperiment: (payload) =>
    request("/api/experiment", { method: "POST", body: JSON.stringify(payload) }),

  listExperiments: () => request("/api/experiments"),

  getExperiment: (filename) => request(`/api/experiments/${filename}`),
};