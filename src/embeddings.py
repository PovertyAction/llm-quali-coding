from __future__ import annotations

import os
from collections.abc import Iterable

from dotenv import load_dotenv
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


# Session 02 helpers: they create their own client, so a script needs one line.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _get_embedding_client() -> OpenAI:
    """Get OpenAI client for embeddings.

    Note: We always use OpenAI for embeddings since Anthropic doesn't
    have a native embeddings API. This works regardless of which LLM
    provider is configured.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required to create embeddings. "
            "Set OPENAI_API_KEY in your .env file."
        )
    return OpenAI(api_key=api_key)


def embed_text(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    """Create a single embedding vector for the provided text.

    Note: Uses OpenAI embeddings regardless of LLM provider setting.
    """
    client = _get_embedding_client()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def embed_texts(
    texts: Iterable[str],
    model: str = DEFAULT_EMBED_MODEL,
) -> list[list[float]]:
    """Create embeddings for a list/iterable of texts.

    NOTE: For simplicity, we call the API once with the list of inputs.
    Note: Uses OpenAI embeddings regardless of LLM provider setting.
    """
    client = _get_embedding_client()
    resp = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in resp.data]
