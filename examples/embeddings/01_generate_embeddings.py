from __future__ import annotations

from src.embeddings import DEFAULT_EMBED_MODEL, embed_text
from src.llm_client import get_provider


def main() -> None:
    """Generate and display a sample text embedding."""
    provider = get_provider()
    print(f"Using LLM provider: {provider}")
    print(f"Using embedding model: {DEFAULT_EMBED_MODEL} (OpenAI)\n")

    text = "Un embedding es una representación numérica del significado de un texto."
    vec = embed_text(text, model=DEFAULT_EMBED_MODEL)

    print("✅ Embedding generado")
    print(f"Dimensions: {len(vec)}")
    print("First 8 values:")
    print(vec[:8])


if __name__ == "__main__":
    main()
