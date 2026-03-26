from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.coding import compute_relevance_scores
from src.embeddings import get_embedding
from src.openai_client import get_client


def split_joint_text(text: str) -> tuple[str, str]:
    r"""Split a joint chunk into moderator question and facilitator responses.

    The joint text format is: '{moderator_question}\\n\\n{facilitator_responses}'
    """
    parts = text.split("\n\n", maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


def main() -> None:
    """Filter chunks by relevance to research question using embeddings."""
    client = get_client()

    # Load chunk embeddings created in step 02
    inp = Path("outputs/01_chunks_with_embeddings.csv")
    if not inp.exists():
        raise FileNotFoundError(
            "Missing outputs/01_chunks_with_embeddings.csv. Run: python examples/02_create_embeddings.py"
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
    kept = df_sorted[df_sorted["question_similarity"] >= threshold].copy()

    # Split joint text into structured columns
    kept[["moderator_question", "responses"]] = kept["text"].apply(
        lambda t: pd.Series(split_joint_text(t))
    )

    print(f"Question: {question}")
    print(f"\n{'=' * 70}")
    print(f"TOP 5 MOST RELEVANT CHUNKS (score >= {threshold})")
    print(f"{'=' * 70}")
    for _, row in kept.head(5).iterrows():
        print(
            f"\n[chunk_id={row['chunk_id']} | score={row['question_similarity']:.3f}]"
        )
        print(f"\nMODERATOR:\n{row['moderator_question']}")
        print(f"\nRESPONSES:\n{row['responses']}")
        print(f"\n{'-' * 70}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "02_relevant_chunks.csv"

    # Save structured columns (most relevant first), drop raw embedding
    cols_to_save = [
        "chunk_id",
        "question_similarity",
        "moderator_question",
        "responses",
    ]
    kept[cols_to_save].to_csv(out_path, index=False)

    print(f"\nKept {len(kept)}/{len(df)} chunks with score >= {threshold}.")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
