from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Callable

import numpy as np
import requests
from rank_bm25 import BM25Okapi


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://172.29.64.1:11434"
).rstrip("/")

OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"
)

OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "120"))
RRF_K = 60
EMBED_BATCH_SIZE = 32

# Each API process maintains its own independent cache.
_INDEX_CACHE = {}


class HybridRetrievalError(RuntimeError):
    """Raised when local hybrid retrieval cannot complete safely."""


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def embed_texts(texts: list[str], prefix: str) -> tuple[list[list[float]], float]:
    """Generate embeddings using the local Ollama embedding endpoint."""
    if not texts:
        return [], 0.0

    start = time.perf_counter()
    embeddings = []

    for offset in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[offset:offset + EMBED_BATCH_SIZE]
        payload = {
            "model": OLLAMA_EMBED_MODEL,
            "input": [f"{prefix}: {text}" for text in batch],
        }

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            batch_embeddings = data.get("embeddings")

            if not isinstance(batch_embeddings, list):
                raise ValueError("Ollama response has no embeddings list.")

            if len(batch_embeddings) != len(batch):
                raise ValueError("Ollama returned an unexpected embedding count.")

            embeddings.extend(batch_embeddings)

        except Exception as error:
            raise HybridRetrievalError(
                f"Local Ollama embedding request failed: {error}"
            ) from error

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return embeddings, elapsed_ms


def dataset_fingerprint(rows: list[dict], text_builder: Callable) -> str:
    content = [text_builder(row) for row in rows]
    raw = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_or_get_index(
    rows: list[dict],
    text_builder: Callable,
    cache_namespace: str,
) -> tuple[dict, bool, float]:
    """Build a service-specific BM25 and semantic index, or reuse its cache."""
    documents = [text_builder(row) for row in rows]
    fingerprint = dataset_fingerprint(rows, text_builder)
    cache_key = f"{cache_namespace}:{OLLAMA_EMBED_MODEL}:{fingerprint}"

    if cache_key in _INDEX_CACHE:
        return _INDEX_CACHE[cache_key], True, 0.0

    start = time.perf_counter()

    tokenized_documents = [tokenize(document) for document in documents]

    if not tokenized_documents or any(not tokens for tokens in tokenized_documents):
        raise HybridRetrievalError("Dataset contains an empty retrieval document.")

    bm25 = BM25Okapi(tokenized_documents)

    corpus_embeddings, embedding_ms = embed_texts(
        documents, "search_document"
    )

    matrix = np.asarray(corpus_embeddings, dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[0] != len(rows):
        raise HybridRetrievalError("Invalid corpus embedding matrix.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / np.maximum(norms, 1e-12)

    index = {
        "fingerprint": fingerprint,
        "bm25": bm25,
        "embeddings": normalized_matrix,
        "documents": documents,
        "record_count": len(rows),
        "embedding_build_ms": embedding_ms,
        "index_build_ms": round(
            (time.perf_counter() - start) * 1000, 2
        ),
    }

    # Remove stale versions for this service/model.
    for old_key in list(_INDEX_CACHE):
        if old_key.startswith(
            f"{cache_namespace}:{OLLAMA_EMBED_MODEL}:"
        ):
            del _INDEX_CACHE[old_key]

    _INDEX_CACHE[cache_key] = index
    return index, False, index["index_build_ms"]


def hybrid_retrieve(
    query: str,
    rows: list[dict],
    text_builder: Callable,
    top_k: int = 5,
    cache_namespace: str = "default",
    identity_field: str = "record_ref",
) -> tuple[list[dict], str, dict]:
    """Retrieve using BM25 + semantic similarity + Reciprocal Rank Fusion."""
    if not query or not query.strip():
        return [], "empty_query", {"reason": "Query is empty."}

    if not rows:
        return [], "empty_dataset", {"reason": "Dataset contains no records."}

    total_start = time.perf_counter()

    index, cache_hit, index_build_ms = build_or_get_index(
        rows, text_builder, cache_namespace
    )

    query_embeddings, query_embedding_ms = embed_texts(
        [query.strip()], "search_query"
    )

    query_vector = np.asarray(query_embeddings[0], dtype=np.float32)

    if query_vector.shape[0] != index["embeddings"].shape[1]:
        raise HybridRetrievalError(
            "Query and corpus embedding dimensions do not match."
        )

    query_norm = np.linalg.norm(query_vector)
    if query_norm <= 1e-12:
        raise HybridRetrievalError("Ollama returned a zero-length query embedding.")

    query_vector = query_vector / query_norm

    # BM25 ranking across the complete service dataset.
    bm25_scores = np.asarray(
        index["bm25"].get_scores(tokenize(query)),
        dtype=np.float64,
    )
    bm25_order = np.argsort(-bm25_scores, kind="stable")

    # Semantic cosine similarity across the complete service dataset.
    semantic_scores = index["embeddings"] @ query_vector
    semantic_order = np.argsort(-semantic_scores, kind="stable")

    bm25_ranks = {
        int(row_index): rank
        for rank, row_index in enumerate(bm25_order, start=1)
    }
    semantic_ranks = {
        int(row_index): rank
        for rank, row_index in enumerate(semantic_order, start=1)
    }

    # Reciprocal Rank Fusion.
    fused = []
    for row_index in range(len(rows)):
        bm25_rank = bm25_ranks[row_index]
        semantic_rank = semantic_ranks[row_index]

        rrf_score = (
            1.0 / (RRF_K + bm25_rank)
            + 1.0 / (RRF_K + semantic_rank)
        )

        fused.append({
            "row_index": row_index,
            "bm25_rank": bm25_rank,
            "bm25_score": round(float(bm25_scores[row_index]), 6),
            "semantic_rank": semantic_rank,
            "semantic_similarity": round(
                float(semantic_scores[row_index]), 6
            ),
            "rrf_score": round(float(rrf_score), 8),
        })

    fused.sort(
        key=lambda item: (
            -item["rrf_score"],
            item["row_index"],
        )
    )

    selected = fused[:top_k]
    records = [rows[item["row_index"]] for item in selected]

    # Diagnostics identify records without copying their full personal data.
    trace = {
        "retrieval_method": "bm25_semantic_rrf",
        "embedding_provider": "local_ollama",
        "embedding_model": OLLAMA_EMBED_MODEL,
        "cache_hit": cache_hit,
        "corpus_record_count": len(rows),
        "returned_record_count": len(records),
        "rrf_k": RRF_K,
        "index_build_ms": round(index_build_ms, 2),
        "query_embedding_ms": query_embedding_ms,
        "total_retrieval_ms": round(
            (time.perf_counter() - total_start) * 1000, 2
        ),
        "rankings": [
            {
                "rank": rank,
                "identity": str(
                    rows[item["row_index"]].get(identity_field, "")
                ),
                "bm25_rank": item["bm25_rank"],
                "bm25_score": item["bm25_score"],
                "semantic_rank": item["semantic_rank"],
                "semantic_similarity": item["semantic_similarity"],
                "rrf_score": item["rrf_score"],
            }
            for rank, item in enumerate(selected, start=1)
        ],
    }

    return records, "hybrid_bm25_semantic_rrf", trace
