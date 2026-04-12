"""
eval/run_eval.py — Automated evaluation script against the 20-question eval set.

Scoring (per question, 10 pts max):
  - Factual accuracy:  0-3  (embedding cosine sim against expected answer)
  - Citation quality:  0-3  (fraction of expected doc_ids cited)
  - Reasoning trace:   0-2  (non-empty traces get 2, else 0)
  - Completeness:      0-2  (answer length heuristic)

Run: python eval/run_eval.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
API_URL  = "http://localhost:8000/chat"
KB_DIR   = Path(__file__).parent.parent
EVAL_PATH = KB_DIR / "eval_set_ai_healthcare.json"
OUT_PATH  = KB_DIR / "eval" / "eval_results.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

_embedder: SentenceTransformer = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ── Scorers ───────────────────────────────────────────────────────────────────

def score_factual(response_answer: str, expected_answer: str) -> float:
    """Embedding cosine similarity → 0-3 pts."""
    emb = get_embedder()
    vecs = emb.encode([response_answer, expected_answer], normalize_embeddings=True)
    sim = float(np.dot(vecs[0], vecs[1]))
    # sim in [-1,1]; map [0,1] → [0,3]
    sim = max(0.0, sim)
    return round(sim * 3, 2)


def score_citations(response_answer: str, expected_doc_ids: list[str]) -> float:
    """Fraction of expected doc_ids cited in the answer → 0-3 pts."""
    if not expected_doc_ids:
        return 3.0
    cited = sum(1 for doc_id in expected_doc_ids if doc_id in response_answer)
    return round((cited / len(expected_doc_ids)) * 3, 2)


def score_reasoning(reasoning_trace: str) -> float:
    """0-2 based on whether a meaningful trace exists."""
    if not reasoning_trace or len(reasoning_trace.strip()) < 30:
        return 0.0
    return 2.0


def score_completeness(response_answer: str, question: str) -> float:
    """0-2 heuristic: answers > 80 words get full marks."""
    words = len(response_answer.split())
    if words >= 80:
        return 2.0
    elif words >= 40:
        return 1.0
    return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eval():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    questions = eval_set["questions"]
    results = []
    total_score = 0.0
    max_score = len(questions) * 10

    print(f"\n{'='*60}")
    print(f"MedRAG Evaluation — {len(questions)} questions")
    print(f"{'='*60}\n")

    for q in questions:
        eval_id   = q["eval_id"]
        question  = q["question"]
        expected  = q["expected_answer"]
        src_docs  = q["source_docs"]
        difficulty = q["difficulty"]

        print(f"[{eval_id}] ({difficulty}) {question[:70]}…")

        try:
            t0 = time.time()
            resp = requests.post(API_URL, json={"query": question, "use_critic": False}, timeout=120)
            elapsed = time.time() - t0

            if resp.status_code != 200:
                print(f"  ❌ API error {resp.status_code}: {resp.text[:200]}")
                results.append({"eval_id": eval_id, "error": resp.text[:200], "score": 0})
                continue

            data = resp.json()
            response_answer = data.get("verified_answer") or data.get("final_answer", "")
            reasoning_trace = data.get("reasoning_trace", "")

            # Score
            s_factual     = score_factual(response_answer, expected)
            s_citations   = score_citations(response_answer, src_docs)
            s_reasoning   = score_reasoning(reasoning_trace)
            s_completeness = score_completeness(response_answer, question)
            total = s_factual + s_citations + s_reasoning + s_completeness

            total_score += total
            print(f"  Score: {total:.1f}/10  "
                  f"[fact={s_factual} cite={s_citations} reason={s_reasoning} complete={s_completeness}]  "
                  f"({elapsed:.1f}s)")

            results.append({
                "eval_id":        eval_id,
                "question":       question,
                "difficulty":     difficulty,
                "expected":       expected,
                "response":       response_answer,
                "reasoning_trace": reasoning_trace,
                "retrieved_docs": [d["doc_id"] for d in data.get("retrieved_docs", [])],
                "score":          round(total, 2),
                "breakdown": {
                    "factual_accuracy": s_factual,
                    "citation_quality": s_citations,
                    "reasoning_trace":  s_reasoning,
                    "completeness":     s_completeness,
                },
                "elapsed_s": round(elapsed, 1),
            })

        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results.append({"eval_id": eval_id, "error": str(e), "score": 0})

    # ── Summary ───────────────────────────────────────────────────────────────
    normalized = round((total_score / max_score) * 50, 1)  # spec: 50 pts total
    print(f"\n{'='*60}")
    print(f"Total raw score:  {total_score:.1f} / {max_score}")
    print(f"Normalized score: {normalized} / 50")

    by_difficulty = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        by_difficulty.setdefault(d, []).append(r.get("score", 0))
    for d, scores in sorted(by_difficulty.items()):
        print(f"  {d}: avg {sum(scores)/len(scores):.1f}/10")
    print(f"{'='*60}\n")

    # Save results
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_raw":    round(total_score, 2),
                "max_raw":      max_score,
                "normalized_50": normalized,
                "by_difficulty": {d: round(sum(s)/len(s), 2) for d, s in by_difficulty.items()},
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUT_PATH}")


if __name__ == "__main__":
    run_eval()
