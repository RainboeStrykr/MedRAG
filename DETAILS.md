# MedRAG — Technical Deep Dive

This document explains, in full detail, how every component of MedRAG works — from raw JSON to a cited, verified answer in the browser.

---

## Table of Contents

1. [Knowledge Base & Data](#1-knowledge-base--data)
2. [Indexing Pipeline](#2-indexing-pipeline-indexerpy)
3. [Retrieval Layer](#3-retrieval-layer-retrieverpy)
4. [Agent 1 — Query Decomposer & Retriever](#4-agent-1--query-decomposer--retriever)
5. [Agent 2 — Reasoner & Synthesizer](#5-agent-2--reasoner--synthesizer)
6. [Agent 3 — Critic](#6-agent-3--critic)
7. [FastAPI Server](#7-fastapi-server-mainpy)
8. [Frontend Architecture](#8-frontend-architecture)
9. [Evaluation Script](#9-evaluation-script)
10. [Design Decisions & Trade-offs](#10-design-decisions--trade-offs)

---

## 1. Knowledge Base & Data

### `knowledge_base_ai_healthcare.json`

Contains **60 documents** across four source types:

| Type | Count |
|---|---|
| Research Paper | 32 |
| Blog | 13 |
| Market Report | 9 |
| Newsletter | 6 |

Each document has the following schema:

```json
{
  "doc_id":      "DOC-001",
  "title":       "Deep Learning for Automated Diabetic Retinopathy Screening at Scale",
  "source_type": "research_paper",
  "source":      "Nature Medicine",
  "url":         "https://...",
  "date":        "2025-08-14",
  "tags":        ["medical imaging", "ophthalmology", "deep learning", "screening"],
  "text":        "Diabetic retinopathy (DR) remains..."
}
```

The texts average ~241 words each (synthetic, modelled on real-world publications). All metadata is preserved through the entire pipeline so every retrieved chunk can be traced back to its source.

### `eval_set_ai_healthcare.json`

Contains **20 evaluation questions** across three difficulty levels:

| Difficulty | Count | Reasoning Type |
|---|---|---|
| Easy | 3 | Factual retrieval (single doc) |
| Medium | 12 | Comparison, evidence synthesis, multi-doc |
| Hard | 5 | Cross-corpus analysis (up to 6 docs) |

Each question includes `expected_answer`, `source_docs` (expected doc IDs), and a `reasoning_type` label. These drive the automated eval script.

---

## 2. Indexing Pipeline (`indexer.py`)

Run once to build the search index. Steps:

### Step 1 — Chunking

```python
CHUNK_SIZE    = 500   # words
CHUNK_OVERLAP = 50    # words
```

Documents are split using a sliding window over whitespace-split words. Since the documents average 241 words each, every document fits within one chunk — meaning the index contains **60 chunks, one per document**. This is intentional: the documents are already dense summaries, and splitting short documents would fragment context rather than improve it.

Each chunk dict contains all metadata from the original document plus a `chunk_id` of the form `DOC-001_chunk_0`.

### Step 2 — Dense Embeddings

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True)
```

`all-MiniLM-L6-v2` produces **384-dimensional sentence embeddings**. Embeddings are L2-normalised before storage, which converts inner-product similarity into cosine similarity.

**Why MiniLM-L6-v2?**
- 384-dim vs 768-dim for larger models → 2× faster retrieval and embedding
- Excellent retrieval quality on STS and BEIR benchmarks relative to its size
- Works entirely on CPU (no GPU needed for 60-document corpus)

### Step 3 — FAISS Index

```python
faiss_index = faiss.IndexFlatIP(dim)   # Inner Product = cosine (after normalisation)
faiss_index.add(embeddings)
```

`IndexFlatIP` performs exact exhaustive search. With only 60 vectors, approximate nearest-neighbour (ANN) methods like HNSW would add complexity with zero benefit — exact search over 60 vectors takes microseconds.

### Step 4 — BM25 Index

```python
tokenized = [re.findall(r"\w+", chunk["text"].lower()) for chunk in chunks]
bm25 = BM25Okapi(tokenized)
```

BM25 (Best Match 25) is a classical probabilistic term-frequency ranking function. It is particularly good at:
- Exact keyword matches (model names, drug names, acronyms like "ABDM", "FAISS")
- Rare terminology that embeddings may not distinguish well  

Both the BM25 index and the tokenized corpus are serialised with `pickle` for fast reloading.

### Output

```
index/
  faiss.index      ← FAISS binary index (384-dim, 60 vectors)
  chunks.pkl       ← list of 60 chunk dicts with full metadata
  bm25.pkl         ← BM25Okapi object
  tokenized.pkl    ← tokenized corpus (for BM25)
```

---

## 3. Retrieval Layer (`retriever.py`)

The retriever implements three functions, all lazy-loading the index on first call:

### `dense_retrieve(query, k)`

1. Embeds the query with the same MiniLM model used at indexing time
2. Calls `faiss_index.search(qvec, k)` → returns `(indices, scores)` 
3. Returns `[(chunk_index, cosine_score), ...]` sorted descending

### `sparse_retrieve(query, k)`

1. Tokenises the query: `re.findall(r"\w+", query.lower())`
2. Calls `bm25.get_scores(tokens)` → BM25 score for every chunk
3. Returns top-k `[(chunk_index, bm25_score), ...]` sorted descending

### `hybrid_retrieve(query, k, rrf_k=60)` — the main interface

**Reciprocal Rank Fusion (RRF):**

```
RRF_score(doc) = Σ  1 / (rrf_k + rank_in_list)
               lists
```

For each result list (dense, sparse), every document at rank `r` receives `1 / (60 + r)`. Scores from both lists are summed. The parameter `rrf_k=60` was chosen because it is the standard default and dampens the influence of very-high-ranked documents from a single modality.

**Deduplication by `doc_id`:** After sorting by RRF score, only the highest-scoring chunk per `doc_id` is kept. This ensures the answer is grounded in diverse documents rather than multiple chunks from the same source.

**Why hybrid?**
- Dense retrieval excels at **paraphrase and semantic similarity** ("what are the risks of…" → retrieves "safety concerns with…")
- Sparse retrieval excels at **exact-match queries** (model names, acronyms, specific statistics)
- RRF fusion outperforms either alone without needing to tune a weighting parameter

---

## 4. Agent 1 — Query Decomposer & Retriever

Implemented in `agents.py · retrieve_context()`.

### Step 0 — Query Routing

```
SYSTEM: Classify the user query as "simple" or "complex". 
        Respond with exactly one word.
```

A fast Qwen3:8b call (`temperature=0.0`) classifies the query:
- **Simple** → single `hybrid_retrieve(query, k=5)` call
- **Complex** → decompose into sub-queries, then retrieve per sub-query

This routing gives a **speed vs depth trade-off**: simple factual questions (e.g., "What is AlphaFold 3?") don't need multi-hop retrieval and can be answered in half the time.

### Step 1 — Query Decomposition (complex only)

```
SYSTEM: Decompose this question into 1-4 focused sub-queries.
        Output ONLY a JSON array of strings.
```

The LLM breaks a multi-part question like:
> *"Compare the approaches of Google, Microsoft, and Amazon in healthcare AI. Which has the highest revenue?"*

into:
```json
[
  "What is Google's healthcare AI strategy and products?",
  "What is Microsoft's healthcare AI strategy and products?",
  "What is Amazon's healthcare AI strategy and products?",
  "Which big tech company has the highest healthcare AI revenue in 2025?"
]
```

A regex extracts the JSON array robustly from the LLM output with a fallback to the original query if parsing fails.

### Step 2 — Per-sub-query Retrieval

Each sub-query runs through `hybrid_retrieve(sq, k=4)`. Results are merged, deduplicating by `doc_id` (first occurrence wins = highest-scoring chunk for that doc). This gives up to `4 sub-queries × 4 docs = 16 candidates`, deduplicated to the top unique documents.

---

## 5. Agent 2 — Reasoner & Synthesizer

### Context Block Construction

Retrieved chunks are formatted as:

```
[DOC-018] FDA's Evolving Framework for AI/ML-Based Software as a Medical Device (STAT Health Tech, 2025-10-12)
<first 1200 chars of chunk text>

---

[DOC-021] The Global Race for AI Healthcare Regulation ...
```

The 1200-character trim prevents context overflow while preserving the most important content (documents average ~1500 chars, so this is a light trim).

### Prompt Design

```
SYSTEM:
  You receive a user question and retrieved document excerpts tagged with [DOC-ID].
  
  Produce TWO sections:
  
  ### REASONING TRACE
  Think step by step. Reference specific document IDs as you reason.
  
  ### FINAL ANSWER
  EVERY factual claim MUST be followed by a [DOC-XXX] citation.
  No free-floating facts without citations.
```

Forcing the model to produce a separate **REASONING TRACE** section before the **FINAL ANSWER** implements Chain-of-Thought (CoT) prompting. This has two effects:
1. The reasoning is **visible** in the UI (can be expanded by the user)
2. The model "thinks before it writes" — consistently higher factual accuracy

### Conversation History

The last 4 exchanges (8 messages) from the conversation are prepended to the messages list. This allows follow-up questions like:
> "You mentioned ABDM earlier — how does it compare to TEFCA in the US?"

without re-stating context.

### Output Parsing

The response is split on the `### FINAL ANSWER` delimiter. If the delimiter is missing (model went off-script), the entire output is treated as the final answer and `reasoning_trace` is set to a fallback string.

---

## 6. Agent 3 — Critic

### Purpose

The Critic acts as a fact-checker over the Reasoner's output. It is independent — it receives the final answer and the source chunks, and verifies each `[DOC-XXX]` citation without knowing what the Reasoner's chain-of-thought said.

### Prompt Design

```
SYSTEM:
  For each [DOC-XXX] citation, verify the claim is supported by that document.
  Flag claims that are NOT supported or have no citation.
  
  ### CRITIQUE
  List each issue, or say "No issues found — all citations verified."
  
  ### VERIFIED ANSWER
  Corrected answer. Mark unsupported claims [UNVERIFIED].
```

`temperature=0.1` (lower than Agent 2's 0.3) — the Critic should be conservative and precise, not creative.

### Output

Returns `(critique, verified_answer)`. The UI shows:
- ✅ **"All citations verified"** — green bar, collapsed by default
- ⚠️ **"Issues flagged"** — amber bar, collapsed, expandable to see specifics

The `verified_answer` replaces the `final_answer` in the UI response.

---

## 7. FastAPI Server (`main.py`)

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns model name |
| `POST` | `/chat` | Full pipeline run |
| `POST` | `/index` | Rebuild index (admin) |

### `/chat` Request / Response

**Request:**
```json
{
  "query": "How does federated learning protect patient privacy?",
  "conversation_history": [
    {"role": "user",      "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for..."}
  ],
  "use_critic": true
}
```

**Response:**
```json
{
  "query_type":      "complex",
  "sub_queries":     ["What is federated learning?", "How does it protect privacy?"],
  "retrieved_docs":  [{"doc_id": "DOC-036", "title": "...", ...}],
  "reasoning_trace": "According to [DOC-036], federated learning...",
  "final_answer":    "Federated learning protects privacy by... [DOC-036]",
  "critique":        "No issues found — all citations verified.",
  "verified_answer": "Federated learning protects privacy by... [DOC-036]"
}
```

### CORS

`allow_origins=["*"]` — permissive for local development. Tighten for production.

---

## 8. Frontend Architecture

### State Model (`page.tsx`)

```typescript
messages: Message[]           // full conversation thread
activeSources: SourceDoc[]    // docs shown in the sidebar (from last AI response)
selectedDoc: SourceDoc | null // currently open doc modal
useCritic: boolean            // toggle agent 3 on/off
loading: boolean              // request in flight
```

### Data Flow

1. User types and presses Enter (or clicks a suggestion card)
2. A **loading message** is optimistically inserted into the thread (shows animated dots)
3. `fetch()` POSTs to `/chat` with the query + last 4 conversation turns
4. On response, the loading message is replaced with the full `Message` object containing the `ChatApiResponse`
5. `activeSources` is updated to the retrieved docs of the most recently clicked message

### Citation Rendering (`FinalAnswer.tsx`)

The answer text is split on `[DOC-XXX]` pattern:
```typescript
text.split(/(\[DOC-\d{3}\])/g)
```

Each matched citation is replaced with a `<span className="citation">` chip. Clicking it calls `onDocClick(doc)` → opens `DocModal` with full metadata.

### Component Breakdown

| Component | Responsibility |
|---|---|
| `page.tsx` | State management, API calls, layout orchestration |
| `ReasoningPanel` | Collapsible `<details>`-style accordion for CoT trace |
| `CriticPanel` | Amber/green collapsible panel for critic output |
| `FinalAnswer` | Splits text on `[DOC-XXX]`, renders clickable citation chips |
| `SourceCard` | One entry in the right sidebar — doc_id, title, source, type badge |
| `DocModal` | Full-screen overlay with doc metadata, URL, tags |

### CSS Design System (`globals.css`)

Fully handwritten vanilla CSS with CSS custom properties:
- **Dark theme** with `--clr-bg: #080c14` base
- **Google Fonts** — Inter for UI, JetBrains Mono for doc IDs and reasoning trace
- **Glassmorphism** header with `backdrop-filter: blur(12px)`
- **Micro-animations** — `fadeIn` on message entry, `dotBounce` loading dots, pulse on status indicator
- **Type-coloured badges** — research_paper (blue), market_report (amber), blog (green), newsletter (violet)

---

## 9. Evaluation Script (`eval/run_eval.py`)

Runs all 20 questions against the live `/chat` API and scores each response on four axes:

### Scoring Breakdown

| Axis | Max | Method |
|---|---|---|
| **Factual Accuracy** | 3 | Cosine similarity between response and expected answer (MiniLM-L6-v2) |
| **Citation Quality** | 3 | Fraction of expected `doc_id`s present in response text |
| **Reasoning Trace** | 2 | Binary: non-empty trace (>30 chars) = 2, else 0 |
| **Completeness** | 2 | Word count heuristic: ≥80 words = 2, ≥40 words = 1, else 0 |
| **Total** | **10** | Per question, normalised to 50 pts total (spec) |

### Output

```json
{
  "summary": {
    "total_raw":      147.5,
    "max_raw":        200,
    "normalized_50":  36.9,
    "by_difficulty":  {"easy": 8.4, "medium": 7.1, "hard": 6.2}
  },
  "results": [...]
}
```

Results are saved to `eval/eval_results.json`. Each result entry includes the question, expected answer, system response, per-axis scores, retrieved doc IDs, and elapsed time.

---

## 10. Design Decisions & Trade-offs

### Why FAISS IndexFlatIP over HNSW?

With 60 documents, exact exhaustive search (`IndexFlatIP`) takes ~0.1ms. HNSW would add ~50KB of graph overhead and graph construction time with no benefit. IndexFlatIP also has zero approximation error — every result is optimal.

### Why one chunk per document?

The documents average 241 words. At a 500-word chunk size, every document is a single chunk. Splitting documents into sub-chunks would:
- Lose cross-sentence context needed for reasoning
- Create fragments that score lower individually, reducing recall
- Add complexity (chunk merging) for no retrieval benefit on this corpus

If the corpus were scaled to documents >1000 words, chunking would be revisited.

### Why RRF over weighted sum?

Weighted sum `(α × dense_score + β × sparse_score)` requires tuning α and β on a validation set. RRF is parameter-free (the `rrf_k=60` default is robust across datasets) and consistently matches or outperforms weighted sum in literature without any dataset-specific tuning.

### Why Qwen3:8b?

- Runs fully locally via Ollama — no API key, no data leaving the machine
- Strong reasoning ability for an 8B model (competitive with Llama-3-8B-Instruct)
- Structured output following (`JSON only`, section headers) is reliable
- Trade-off: ~60–100s per query on CPU vs ~2–5s for a cloud API

### Why `all-MiniLM-L6-v2` over larger models?

- 384-dim embeddings: index is 60 × 384 × 4 bytes = ~90KB (trivial)
- Indexing 60 documents takes <2 seconds
- Retrieval quality on medical text is sufficient — the LLM handles semantic nuance in the reasoning step
- No GPU required

### Why Vanilla CSS over Tailwind?

Vanilla CSS gives full control over the design, avoids the Tailwind JIT build step, and keeps the bundle smaller. The design system is defined entirely in CSS custom properties (`--clr-*`, `--radius-*`, `--trans-*`) making theming consistent.

### Citation verification vs hallucination

The Critic Agent (Agent 3) reduces hallucinated citations by having a second LLM pass verify that each `[DOC-XXX]` claim is actually supported by the retrieved text. It cannot catch hallucinations about facts not present in any retrieved document, but it does catch the most common failure mode: the model citing the wrong document for a statistic.