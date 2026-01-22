from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.coding import compute_relevance_scores, filter_relevant
from src.embeddings import get_embedding
from src.openai_client import get_client


def main() -> None:
    """Filter chunks by relevance to research question using embeddings."""
    client = get_client()

    # Load chunk embeddings created in step 02
    inp = Path("outputs/01_chunks_with_embeddings.csv")
    if not inp.exists():
        raise FileNotFoundError(
            "Missing outputs/03_chunks_with_embeddings.csv. Run: python examples/02_create_embeddings.py"
        )

    df = pd.read_csv(inp)
    df["embedding"] = df["embedding"].apply(json.loads)

    question = "What helped facilitators integrate Bloom with Love into existing family services?"
    q_emb = get_embedding(client, question)

    df = compute_relevance_scores(df, q_emb)
    df_sorted = df.sort_values("question_similarity", ascending=False).reset_index(
        drop=True
    )

    # Filter by relevance threshold
    threshold = 0.20
    kept = df_sorted[df_sorted["question_similarity"] >= threshold]

    print("Question:")
    print(question)
    print("\n--- Top 5 (most relevant) ---")
    for _, row in kept.head(5).iterrows():
        print(f"score={row['question_similarity']:.3f} | chunk_id={row['chunk_id']}")
        print(row["text"][:300] + "..." if len(row["text"]) > 300 else row["text"])
        print()

    rel = filter_relevant(df_sorted, threshold=threshold)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "02_relevant_chunks.csv"

    rel_to_save = rel.copy()
    rel_to_save["embedding"] = rel_to_save["embedding"].apply(json.dumps)
    rel_to_save.to_csv(out_path, index=False)

    print(f"\nKept {len(kept)}/{len(df)} chunks with score >= {threshold}.")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
