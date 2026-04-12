"""
agents.py — Multi-agent pipeline: Decomposer → Reasoner → Critic
Uses local Qwen3:8b via Ollama.
"""

import json
import re
from typing import Any

import ollama

from retriever import hybrid_retrieve, get_chunk_by_doc_id

MODEL = "qwen3:8b"

# ─── Ollama helper ────────────────────────────────────────────────────────────

def _chat(messages: list[dict], temperature: float = 0.2) -> str:
    """Call Ollama and return the assistant text content."""
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": temperature, "num_predict": 2048},
    )
    return response["message"]["content"].strip()


# ─── Query complexity classifier ─────────────────────────────────────────────

CLASSIFY_SYSTEM = """You are a query router for a medical AI knowledge base.
Classify the user query as either:
- "simple": a single factual question answered by one document
- "complex": requires synthesizing multiple documents or multi-hop reasoning

Respond with exactly one word: simple OR complex."""

def classify_query(query: str) -> str:
    result = _chat([
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user",   "content": query},
    ], temperature=0.0)
    return "simple" if "simple" in result.lower() else "complex"


# ─── Agent 1: Query Decomposer & Retriever ───────────────────────────────────

DECOMPOSE_SYSTEM = """You are Agent 1 of a medical AI research assistant.
Your job: decompose a user question into 1-4 focused sub-queries that together cover the full question.
Each sub-query should target a specific fact, comparison, or concept.

Output ONLY a JSON array of strings. No markdown, no explanation.
Example: ["What is AlphaFold 3?", "How does AlphaFold 3 impact drug discovery timelines?"]"""

def decompose_query(query: str) -> list[str]:
    result = _chat([
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user",   "content": f"Decompose this question into sub-queries:\n{query}"},
    ], temperature=0.1)
    # Extract JSON array robustly
    match = re.search(r'\[.*?\]', result, re.DOTALL)
    if match:
        try:
            sub_queries = json.loads(match.group())
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                return sub_queries[:4]
        except json.JSONDecodeError:
            pass
    # Fallback: use original query
    return [query]


def retrieve_context(query: str, is_simple: bool) -> tuple[list[dict], list[str]]:
    """
    Retrieves relevant chunks.
    Simple: single hybrid_retrieve call (k=5)
    Complex: decompose + retrieve per sub-query, merge unique docs
    """
    if is_simple:
        sub_queries = [query]
        k_per_query = 5
    else:
        sub_queries = decompose_query(query)
        k_per_query = 4

    seen_doc_ids: set[str] = set()
    retrieved_docs: list[dict] = []

    for sq in sub_queries:
        results = hybrid_retrieve(sq, k=k_per_query)
        for r in results:
            if r["doc_id"] not in seen_doc_ids:
                retrieved_docs.append(r)
                seen_doc_ids.add(r["doc_id"])

    return retrieved_docs, sub_queries


# ─── Agent 2: Reasoner & Synthesizer ─────────────────────────────────────────

REASON_SYSTEM = """You are Agent 2 of a medical AI research assistant.
You receive a user question and a set of retrieved document excerpts, each tagged with a DOC-ID.

Your task — produce a structured response with TWO sections:

### REASONING TRACE
Think step by step. Reference specific document IDs as you reason.
For example: "According to [DOC-003], the RCT showed ..."
Be thorough: identify which docs are most relevant, what they say, and how they connect.

### FINAL ANSWER
Write a clear, complete answer to the user's question.
EVERY factual claim MUST be followed by a citation in the form [DOC-XXX].
No free-floating facts without citations. Synthesize across documents when needed.

Use plain text with [DOC-XXX] inline citations. Do not use bullet points purely; mix prose and bullets where appropriate."""

def build_context_block(retrieved_docs: list[dict]) -> str:
    parts = []
    for doc in retrieved_docs:
        parts.append(
            f"[{doc['doc_id']}] {doc['title']} ({doc['source']}, {doc['date']})\n"
            f"{doc['text'][:1200]}"  # trim very long chunks
        )
    return "\n\n---\n\n".join(parts)


def reason_and_synthesize(
    query: str,
    retrieved_docs: list[dict],
    conversation_history: list[dict],
) -> tuple[str, str]:
    """
    Returns (reasoning_trace, final_answer).
    """
    context_block = build_context_block(retrieved_docs)
    
    # Build messages with conversation history
    messages = [{"role": "system", "content": REASON_SYSTEM}]
    
    # Add prior turns (last 4 exchanges)
    for turn in conversation_history[-4:]:
        messages.append(turn)

    user_content = (
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        f"USER QUESTION: {query}"
    )
    messages.append({"role": "user", "content": user_content})

    raw = _chat(messages, temperature=0.3)

    # Parse sections
    reasoning_trace = ""
    final_answer = ""

    if "### FINAL ANSWER" in raw:
        parts = raw.split("### FINAL ANSWER", 1)
        reasoning_part = parts[0]
        final_part = parts[1].strip()
        reasoning_trace = reasoning_part.replace("### REASONING TRACE", "").strip()
        final_answer = final_part
    else:
        # Fallback: treat everything as final answer
        final_answer = raw
        reasoning_trace = "No explicit reasoning trace produced."

    return reasoning_trace, final_answer


# ─── Agent 3: Critic ─────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You are Agent 3 — the Critic and Verifier of a medical AI research assistant.

You receive:
1. A final answer with inline [DOC-XXX] citations
2. The source document excerpts

Your task:
- For each [DOC-XXX] citation, verify the claim is actually supported by that document
- Flag any claim that is NOT supported (hallucinated or wrong doc cited)
- Flag any factual claim WITHOUT a citation

Output TWO sections:

### CRITIQUE
List each issue you find, or say "No issues found — all citations verified."
Format: "CLAIM: '...' ISSUE: ..."

### VERIFIED ANSWER
The corrected answer. Keep supported claims unchanged. Remove or mark [UNVERIFIED] for unsupported ones."""

def critique_answer(
    final_answer: str,
    retrieved_docs: list[dict],
) -> tuple[str, str]:
    """
    Returns (critique, verified_answer).
    """
    context_block = build_context_block(retrieved_docs)
    
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": (
            f"SOURCE DOCUMENTS:\n{context_block}\n\n"
            f"ANSWER TO VERIFY:\n{final_answer}"
        )},
    ]

    raw = _chat(messages, temperature=0.1)

    critique = ""
    verified_answer = ""

    if "### VERIFIED ANSWER" in raw:
        parts = raw.split("### VERIFIED ANSWER", 1)
        critique = parts[0].replace("### CRITIQUE", "").strip()
        verified_answer = parts[1].strip()
    else:
        critique = raw
        verified_answer = final_answer  # fallback: keep original

    return critique, verified_answer


# ─── Master pipeline ──────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    conversation_history: list[dict],
    use_critic: bool = True,
) -> dict[str, Any]:
    """
    Full MedRAG pipeline.
    Returns dict with all intermediate and final outputs.
    """
    # Step 0: classify
    query_type = classify_query(query)
    is_simple = (query_type == "simple")

    # Step 1: retrieve
    retrieved_docs, sub_queries = retrieve_context(query, is_simple)

    # Step 2: reason
    reasoning_trace, final_answer = reason_and_synthesize(
        query, retrieved_docs, conversation_history
    )

    # Step 3: critic (optional)
    critique = None
    verified_answer = final_answer
    if use_critic:
        critique, verified_answer = critique_answer(final_answer, retrieved_docs)

    # Build source metadata list (unique docs, no chunk text)
    sources = [
        {
            "doc_id":      d["doc_id"],
            "title":       d["title"],
            "source":      d["source"],
            "source_type": d["source_type"],
            "url":         d["url"],
            "date":        d["date"],
            "tags":        d["tags"],
        }
        for d in retrieved_docs
    ]

    return {
        "query_type":      query_type,
        "sub_queries":     sub_queries,
        "retrieved_docs":  sources,
        "reasoning_trace": reasoning_trace,
        "final_answer":    final_answer,
        "critique":        critique,
        "verified_answer": verified_answer,
    }
