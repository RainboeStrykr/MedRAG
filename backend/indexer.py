"""
indexer.py — Builds the FAISS + BM25 index from the knowledge base JSON.
Run once: python indexer.py
"""

import json
import os
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
KB_PATH = Path(__file__).parent.parent / "knowledge_base_ai_healthcare.json"
INDEX_DIR = Path(__file__).parent / "index"

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500      # tokens (approx chars / 4)
CHUNK_OVERLAP = 50    # tokens overlap

# ── Tokeniser (simple whitespace) ──────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

# ── Chunker ────────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += size - overlap
    return chunks

# ── Main ────────────────────────────────────────────────────────────────────────
def build_index():
    print("Loading knowledge base …")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    documents = kb["documents"]
    
    # ── 1. Chunk documents ────────────────────────────────────────────────────
    print("Chunking documents …")
    chunks = []          # list of dicts with text + metadata
    for doc in tqdm(documents):
        doc_chunks = chunk_text(doc["text"])
        for i, chunk_text_str in enumerate(doc_chunks):
            chunks.append({
                "chunk_id":    f"{doc['doc_id']}_chunk_{i}",
                "doc_id":      doc["doc_id"],
                "title":       doc["title"],
                "source":      doc["source"],
                "source_type": doc["source_type"],
                "url":         doc["url"],
                "date":        doc["date"],
                "tags":        doc["tags"],
                "text":        chunk_text_str,
            })

    print(f"  → {len(chunks)} total chunks from {len(documents)} documents")

    # ── 2. Dense embeddings ────────────────────────────────────────────────────
    print(f"Embedding with {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # ── 3. FAISS index (Inner Product = cosine because embeddings normalised) ──
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    # ── 4. BM25 index ─────────────────────────────────────────────────────────
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # ── 5. Save to disk ───────────────────────────────────────────────────────
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(faiss_index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(INDEX_DIR / "tokenized.pkl", "wb") as f:
        pickle.dump(tokenized, f)

    print(f"\n✅ Index saved to {INDEX_DIR}")
    print(f"   FAISS dim={dim}, vectors={faiss_index.ntotal}")
    print(f"   Chunks: {len(chunks)}")


if __name__ == "__main__":
    build_index()
