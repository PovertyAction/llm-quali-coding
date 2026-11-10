"""Embeddings from an open-weights model that runs on the laptop.

Nothing is sent to an API. The first call downloads the model (about 1.1 GB)
to the Hugging Face cache; after that it works offline.
"""

from __future__ import annotations

from functools import cache

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@cache
def _load(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """Embed texts with a local model.

    Vectors are normalized to length 1, so the dot product in
    `src.similarity.dot_similarity` equals cosine similarity.
    """
    vectors = _load(model_name).encode(texts, normalize_embeddings=True)
    return vectors.tolist()
