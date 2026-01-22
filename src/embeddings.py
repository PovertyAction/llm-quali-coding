from __future__ import annotations

from openai import OpenAI

from .openai_client import load_config


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Create a single embedding vector for the given text."""
    cfg = load_config()
    # Embeddings endpoint takes a plain string (no roles/messages)
    response = client.embeddings.create(
        model=cfg.embedding_model,
        input=text,
    )
    return response.data[0].embedding
