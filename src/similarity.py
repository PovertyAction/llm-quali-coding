from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def dot_similarity(a: list[float], b: list[float]) -> float:
    """Dot product similarity.

    Many embedding models return normalized vectors, making dot product equivalent to cosine similarity.
    """
    return float(np.dot(a, b))


def top_k_similar(
    query_embedding: list[float],
    items: Iterable[tuple[str, list[float]]],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Return top-k items by similarity.

    items: iterable of (item_id, embedding)
    returns: list of (item_id, score) sorted ascending → descending
    """
    scored = [(item_id, dot_similarity(query_embedding, emb)) for item_id, emb in items]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
