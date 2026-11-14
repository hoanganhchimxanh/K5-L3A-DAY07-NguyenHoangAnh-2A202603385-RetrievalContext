"""Retrieval methods for Day 7: Lexical (TF-IDF), Dense (Embeddings), and Hybrid (RRF).

Implements the Retriever protocol specified in starter/agentic.py.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path to allow running both as module and standalone script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.embeddings import MyEmbedder
except ImportError:
    from embeddings import MyEmbedder


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text into lowercase word tokens."""
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def matches_filter(chunk: dict[str, Any], filters: Optional[dict[str, Any]]) -> bool:
    """Check if a chunk matches given metadata filters (e.g. chapter)."""
    if not filters:
        return True
    meta = chunk.get('metadata', {})
    for key, value in filters.items():
        chunk_val = meta.get(key) or chunk.get(key)
        if chunk_val is None:
            return False
        if str(value).lower() not in str(chunk_val).lower():
            return False
    return True


class TfidfRetriever:
    """Lexical retrieval using TF-IDF vector space model."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in self.counts for term in row)
        total = len(chunks)
        self.idf = {term: math.log((1 + total) / (1 + count)) + 1.0 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in self.counts]

    def _vector(self, row: Counter) -> dict[str, float]:
        norm = math.sqrt(sum((count * self.idf[term]) ** 2 for term, count in row.items()))
        if norm == 0:
            return {}
        return {term: (count * self.idf[term]) / norm for term, count in row.items()}

    def search(self, query: str, filters: Optional[dict[str, Any]] = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        q_row = Counter(query_terms)
        q_norm = math.sqrt(sum((q_row[term] * self.idf.get(term, 1.0)) ** 2 for term in q_row))
        if q_norm == 0:
            return []
        q_vec = {t: (q_row[t] * self.idf.get(t, 1.0)) / q_norm for t in q_row}

        scored = []
        for i, doc_vec in enumerate(self.vectors):
            chunk = self.chunks[i]
            if filters and not matches_filter(chunk, filters):
                continue
            score = sum(val * doc_vec.get(t, 0.0) for t, val in q_vec.items())
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            item = dict(chunk)
            item['score'] = float(score)
            item['rank'] = rank
            results.append(item)
        return results


class DenseRetriever:
    """Dense vector retrieval using query embeddings and cached document embeddings."""

    def __init__(self, chunks: list[dict[str, Any]], embeddings_path: Path | str, embedder: Optional[MyEmbedder] = None) -> None:
        self.chunks = chunks
        self.chunks_by_id = {c['id']: c for c in chunks}
        self.embedder = embedder or MyEmbedder()
        
        emb_path = Path(embeddings_path)
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        with open(emb_path, 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)

    def search(self, query: str, filters: Optional[dict[str, Any]] = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = self.embedder.embed_query(query)
        scored = []
        for chunk in self.chunks:
            cid = chunk['id']
            if cid not in self.vectors:
                continue
            if filters and not matches_filter(chunk, filters):
                continue
            doc_vec = self.vectors[cid]
            score = cosine_similarity(query_vec, doc_vec)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            item = dict(chunk)
            item['score'] = float(score)
            item['rank'] = rank
            results.append(item)
        return results


class HybridRetriever:
    """Hybrid retrieval combining Dense and Lexical results using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense_retriever: DenseRetriever, lexical_retriever: TfidfRetriever, rrf_k: int = 60) -> None:
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, filters: Optional[dict[str, Any]] = None, top_k: int = 3) -> list[dict[str, Any]]:
        candidate_depth = max(top_k * 4, 20)
        dense_results = self.dense.search(query, filters=filters, top_k=candidate_depth)
        lexical_results = self.lexical.search(query, filters=filters, top_k=candidate_depth)

        scores: dict[str, float] = {}
        chunks_map: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            cid = item['id']
            chunks_map[cid] = item
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, item in enumerate(lexical_results, start=1):
            cid = item['id']
            chunks_map[cid] = item
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (cid, score) in enumerate(ranked[:top_k], start=1):
            item = dict(chunks_map[cid])
            item['score'] = float(score)
            item['rank'] = rank
            results.append(item)
        return results


if __name__ == '__main__':
    print("Testing retrievers module self-check...")
    chunks_file = PROJECT_ROOT / 'data' / 'chunks.json'
    if chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as f:
            sample_chunks = json.load(f)[:5]
        tfidf = TfidfRetriever(sample_chunks)
        res = tfidf.search("Inca", top_k=2)
        print(f"Self-check passed! TF-IDF returned {len(res)} results.")
    else:
        print("data/chunks.json not found, module imported successfully.")

