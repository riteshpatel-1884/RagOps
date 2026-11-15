# RAG Ops: measuring every change to a RAG system

A retrieval-augmented generation (RAG) system over a 27-page company handbook, built as three controlled experiments. Each version changes a small number of settings, is run on the same 105 questions and is scored by the same judge, so every improvement (and every regression) is measured rather than assumed.

**Result in one line:** across three versions, context relevancy rose from 0.11 to 0.27 and cost per query fell 34%, with answer quality flat and hallucination at 0%. The trade-off was a small drop on multi-hop questions.

---

## Why this project

Most RAG demos show that a pipeline works. This one asks what a change actually did. For each version I record what changed, what improved, what got worse and how confident the numbers are.

## Dataset

| | |
|---|---|
| Document | `Asterion_Employee_Handbook.pdf` (27 pages) |
| Questions | 105: 40 single-fact, 25 multi-part, 20 multi-hop, 20 unanswerable |
| Ground truth | Every answerable question has an expected answer and verbatim evidence from a specific page (checked by `cli validate`) |
| Judged subset | Judge metrics run on the same 43 of the 85 answerable questions (every second one) for every version |

## Pipeline

PDF → text blocks (PyMuPDF, headers and footers removed) → chunks → embeddings (`gemini-embedding-001`, 768 dimensions) → Qdrant (cosine) → retrieval → answer generation (Groq `openai/gpt-oss-120b`, temperature 0) → evaluation (DeepEval judge, same model for all versions).

## The three versions

| | v1 baseline | v2 smaller chunks | v3 section chunks + dynamic top-k |
|---|---|---|---|
| Chunking | Static, 2,000 chars | Static, 1,000 chars | One chunk per numbered section, at most 800 chars, embedded with its section path |
| Chunks in index | 41 | 70 | 124 |
| Retrieval | Vector, top 3 | Vector, top 4 | Vector, fetch 8, keep hits within 0.08 of the best score, between 2 and 4 chunks |
| Embedding, generator, judge | same | same | same |

**v1** is the baseline: page-sized chunks, plain vector search.

**v2** changed only chunk size and top-k. Multi-part and multi-hop questions were answered more often.

**v3** targets context relevancy. Chunks follow the handbook's headings (`7.3`, `12.1`, ...), so a chunk is a coherent section instead of an arbitrary slice. The section path (for example `12. Leave > 12.2 Carry-forward`) is added to the text that is embedded. At query time, the number of chunks sent depends on how close each hit's score is to the best one, so easy questions get less text.

## Results

Judge metrics: 43 answerable questions. Retrieval and reliability metrics: all 105 questions.

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Answer correctness | 0.893 | 0.914 | 0.907 |
| Faithfulness | 1.000 | 0.996 | 1.000 |
| Answer relevancy | 0.989 | 0.993 | 0.977 |
| Context precision | 0.915 | 0.906 | 0.947 |
| Context recall | 0.977 | 1.000 | 0.977 |
| **Context relevancy** | 0.107 | 0.146 | **0.267** |
| Hit rate @3 | 0.953 | 0.953 | 0.976 |
| False refusal rate (answerable) | 5.9% | 2.4% | 3.5% |
| Hallucination rate (unanswerable) | 0% | 0% | 0% |
| Cost per query (USD) | 0.000249 | 0.000220 | 0.000165 |

Multi-hop questions (n = 20):

| | v1 | v2 | v3 |
|---|---|---|---|
| False refusal | 20% | 10% | 15% |
| Correctness | 0.75 | 0.84 | 0.80 |

### What I learned

- **Smaller chunks fixed multi-hop (v2).** Page-sized chunks diluted the evidence. Smaller chunks plus one more retrieved chunk cut multi-hop false refusals from 20% to 10%.
- **Context relevancy mostly measures context size (v3).** Sending about 3,000 characters to answer a one-line question guarantees a low score. Cutting the text roughly in half almost doubled the metric (0.15 to 0.27) without hurting faithfulness or hallucination.
- **There was a trade-off.** Fewer chunks sometimes means the second piece of evidence for a multi-hop question is not sent: false refusal rose from 10% to 15% and context recall fell from 1.00 to 0.98 (one question).
- **A reranker did not help here.** In an offline retrieval test, a small cross-encoder reranker (FlashRank) lowered the share of multi-part questions where all evidence was retrieved (0.775 vs 0.875), so I did not use it.
- **I picked v3's margin with an offline test, not the judge.** `retrieval_check` compares setups with no LLM calls. Margin 0.05 lost evidence; 0.08 kept the hit rate at v2's level (0.988) while sending 54% less text (1,383 vs 3,034 characters).

| Offline retrieval check | any-hit | all-hit | chars sent | chunks |
|---|---|---|---|---|
| v1 (2000, top 3) | 0.953 | 0.800 | 4001 | 3.0 |
| v2 (1000, top 4) | 0.988 | 0.875 | 3034 | 4.0 |
| section 800, margin 0.03 | 0.941 | 0.825 | 1126 | 2.3 |
| section 800, margin 0.05 | 0.976 | 0.825 | 1212 | 2.5 |
| **section 800, margin 0.08 (v3)** | **0.988** | 0.850 | **1383** | 2.9 |
| section 600, margin 0.05 | 0.988 | 0.775 | 1010 | 2.4 |
| section 800, fixed top 3 | 0.976 | 0.800 | 1380 | 3.0 |

## Metrics

| Metric | What it tells you |
|---|---|
| Faithfulness | Is the answer supported by the retrieved context? |
| Answer relevancy | Does the answer address the question? |
| Answer correctness | Does the answer match the expected answer? |
| Context precision | Are the relevant chunks ranked above irrelevant ones? |
| Context recall | Does the retrieved context contain what is needed? |
| Context relevancy | What share of the retrieved text is relevant to the question? |
| Hit rate @3 | Judge-free: is a gold evidence snippet in the first 3 retrieved chunks? |
| False refusal rate | Answerable questions the system wrongly refused |
| Hallucination rate | Unanswerable questions the system answered anyway |
| Abstention accuracy | Unanswerable questions correctly refused |

Confidence intervals are 95% bootstrap intervals (2,000 resamples, fixed seed).

## Limitations

- **Small judged sample.** Judge metrics use 43 questions, so intervals are about ±0.1. Most differences between versions, other than the context relevancy gain, are within the noise. A one-question difference on a 20-question category moves a rate by 5 points.
- **Tuned and scored on the same questions.** There is no held-out test set, so the results are likely somewhat optimistic.
- **Several changes at once.** v2 changed chunk size and top-k together; v3 changed chunking and retrieval together. I cannot say which change caused which effect.
- **One document.** The section chunker relies on this handbook's numbered headings. It falls back to static chunking if few headings are found, but I have not tested other layouts.
- **Latency is not comparable.** The versions ran on different days under different API load, so latency differences are mostly load, not setup. The cost differences are real, since they come from fewer input tokens.
- **One judge model.** Judge and generator are the same model family, which can bias scores.
- **Hit rate @3** looks only at the first 3 chunks, so it understates v2 (4 chunks) and v3 (up to 4).

## Next steps

- Raise v3's `min_k` to 3 to win back multi-hop questions, and measure the cost in context relevancy.
- Add a separate held-out question set.
- Try a reranker used as a filter (drop low-scoring chunks) rather than as a reorderer.
- Run the same pipeline on a second document.

## Project structure

```
backend/
  app/
    main.py                  FastAPI app: POST a question, choose version v1 | v2 | v3
    versions/
      eval_dataset.jsonl     the 105 questions (shared by all versions)
      cli.py                 validate | check | index | run | judge | summary | save | compare
      toolkit/               config, corpus (chunking), retrieval, vectorstore, embeddings, judge, metrics
      v1/ v2/ v3/            one folder per version: agent, indexing, results and evaluated results
    evaluation/
      retrieval_check.py     compare retrieval setups offline, with no LLM calls
  pdfs/                      the handbook PDF
dashboard/                   results dashboard (lib/experiments.js holds the recorded numbers)
```

> Adjust the paths above to match your repository.

## Setup

Requirements: Python 3.10+, a Groq API key, a Gemini API key and a Qdrant instance (cloud, or `QDRANT_URL=:memory:` for a throw-away local index).

```bash
cd backend
pip install -r requirements.txt
```

Create `backend/.env`:

```
GROQ_API_KEY=...
GEMINI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

Put the handbook PDF in `backend/pdfs/`.

## Reproduce the experiments

Run from `backend/`:

```bash
python -m app.versions.cli validate            # check the dataset against the PDF
python -m app.evaluation.retrieval_check       # offline retrieval comparison (no Groq calls)

python -m app.versions.cli run v3              # answer all 105 questions (resumes if interrupted)
python -m app.versions.cli judge v3 --every 2 --sleep 10   # DeepEval judge
python -m app.versions.cli summary v3          # print the metrics with confidence intervals
python -m app.versions.cli save v3             # write experiment_v3.json
python -m app.versions.cli compare             # side-by-side table of saved versions
```

Use the same `--every 2` for every version so they are judged on the same 43 questions. Embeddings are cached on disk, so re-running costs no embedding calls.

## Tech stack

Python, PyMuPDF, Gemini embeddings, Qdrant, Groq (`openai/gpt-oss-120b`), DeepEval, FastAPI, rank-bm25 and FlashRank (tested offline), React dashboard.