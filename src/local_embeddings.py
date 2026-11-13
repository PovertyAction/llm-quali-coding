"""Embeddings from an open-weights model that runs on the laptop.

Nothing is sent to an API. The first call downloads the model (about 210 MB)
to the Hugging Face cache; after that it works offline.
"""

from __future__ import annotations

import os
from functools import cache

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Set LOCAL_EMBEDDING_MODEL in .env to try another Hugging Face model. Every model
# scores on its own scale, so retune THRESHOLD in
# examples/local/02_filter_classify.py when you change it.
DEFAULT_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "ibm-granite/granite-embedding-97m-multilingual-r2"
)


@cache
def _load(model_name: str) -> SentenceTransformer:
    # float32 gives the same scores on every laptop CPU. Some models ship in
    # bfloat16, which is slower on CPU and shifts scores slightly.
    return SentenceTransformer(model_name, model_kwargs={"dtype": torch.float32})


def embed_texts(
    texts: list[str], model_name: str = DEFAULT_MODEL, *, query: bool = False
) -> list[list[float]]:
    """Embed texts with a local model.

    Set `query=True` for the search side (questions, theme definitions) and
    leave it False for the passages being searched. Models trained with
    separate query and passage prompts need this; others ignore it.

    Vectors are normalized to length 1, so the dot product in
    `src.similarity.dot_similarity` equals cosine similarity.
    """
    model = _load(model_name)
    encode = model.encode_query if query else model.encode_document
    return encode(texts, normalize_embeddings=True).tolist()
