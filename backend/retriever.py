"""
retriever.py — Hybrid dense + sparse retrieval with RRF fusion.
"""

import pickle
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent / "index"
EMBED_MODEL = "all-MiniLM-L6-v2"

_model: Optional[SentenceTransformer] = None
_faiss_index = None
_chunks: list[dict] = []
_bm25 = None
_tokenized: list[list[str]] = []


def _load():
    global _model, _faiss_index, _chunks, _bm25, _tokenized
    if _faiss_index is not None:
        return
    _model = SentenceTransformer(EMBED_MODEL)
    _faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        _chunks = pickle.load(f)
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        _bm25 = pickle.load(f)
    with open(INDEX_DIR / "tokenized.pkl", "rb") as f:
        _tokenized = pickle.load(f)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def dense_retrieve(query: str, k: int = 8) -> list[tuple[int, float]]:
    """Returns (chunk_index, score) sorted by score desc."""
    _load()
    qvec = _model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _faiss_index.search(qvec, k)
    return list(zip(indices[0].tolist(), scores[0].tolist()))


def sparse_retrieve(query: str, k: int = 8) -> list[tuple[int, float]]:
    """Returns (chunk_index, score) sorted by score desc."""
    _load()
    tokens = tokenize(query)
    scores = _bm25.get_scores(tokens)
    top_k = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_k]


def hybrid_retrieve(query: str, k: int = 6, rrf_k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion of dense + sparse results.
    Returns top-k unique chunks (full metadata dicts).
    """
    _load()

    dense_results = dense_retrieve(query, k=k * 2)
    sparse_results = sparse_retrieve(query, k=k * 2)

    rrf_scores: dict[int, float] = {}

    for rank, (idx, _) in enumerate(dense_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, (idx, _) in enumerate(sparse_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

    # Deduplicate by doc_id — prefer highest-scoring chunk per doc
    seen_doc_ids: set[str] = set()
    results = []
    for idx in sorted_indices:
        chunk = _chunks[idx]
        doc_id = chunk["doc_id"]
        if doc_id not in seen_doc_ids:
            results.append({**chunk, "rrf_score": rrf_scores[idx]})
            seen_doc_ids.add(doc_id)
        if len(results) >= k:
            break

    return results


def get_chunk_by_doc_id(doc_id: str) -> Optional[dict]:
    """Get first chunk for a given doc_id (for citation verification)."""
    _load()
    for c in _chunks:
        if c["doc_id"] == doc_id:
            return c
    return None
