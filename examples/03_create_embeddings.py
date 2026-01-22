from __future__ import annotations

import json
from pathlib import Path

from src.chunking import make_chunks
from src.coding import build_chunk_dataframe, embed_chunks
from src.embeddings import get_embedding
from src.openai_client import get_client


def main() -> None:
    """Create embeddings for transcript chunks and save to CSV."""
    client = get_client()

    # Use the English sample transcript (either the hand-provided sample or output of step 02)
    default_inp = Path("data/sample_transcripts/sample_english.md")
    translated = Path("outputs/02_translated_english.md")
    inp = translated if translated.exists() else default_inp

    text = inp.read_text(encoding="utf-8")

    chunks = make_chunks(text, min_chars=250)
    df = build_chunk_dataframe(chunks)

    df = embed_chunks(client, df, text_col="text")

    # Save embeddings as JSON strings (keeps this repo lightweight and dependency-free)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "03_chunks_with_embeddings.csv"

    df_to_save = df.copy()
    df_to_save["embedding"] = df_to_save["embedding"].apply(json.dumps)
    df_to_save.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print("Rows:", len(df_to_save))

    # Tiny demo: compare two short strings
    a = get_embedding(client, "Queen")
    b = get_embedding(client, "King")
    c = get_embedding(client, "Physics")

    import numpy as np

    print("\nDot-product similarities (higher = more semantically similar):")
    print("Queen vs King:", float(np.dot(a, b)))
    print("Queen vs Physics:", float(np.dot(a, c)))


if __name__ == "__main__":
    main()
