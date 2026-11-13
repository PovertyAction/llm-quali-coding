"""Step 2: find relevant passages and code them with local embeddings.

Uses an open-weights embedding model on the laptop. A passage's relevance is
its similarity to the closest codebook theme. Passages below the threshold are
coded "Fuera de tema"; the rest get the most similar theme.

Writes outputs/local/02_coded.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.coding import (
    Theme,
    add_theme_similarity_columns,
    classify_by_max_theme,
    load_themes,
)
from src.local_embeddings import embed_texts

OUT = Path("outputs/local")
# Tuned for the default model in src/local_embeddings.py, whose scores all
# fall between about 0.70 and 0.90. Another model needs another value: step 4
# (just local-compare) prints the agreement for each candidate threshold.
THRESHOLD = 0.78
OFF_TOPIC = "Fuera de tema"


def main() -> None:
    """Score relevance, classify by most similar theme, and save."""
    passages = OUT / "01_passages_deid.csv"
    if not passages.exists():
        raise SystemExit(f"Missing {passages}. Run step 1 first: just local-deid")
    df = pd.read_csv(passages)
    df["embedding"] = embed_texts(df["text"].tolist())

    themes = load_themes(Path("data/fictitious/codebook_ficticio.json"))
    vectors = embed_texts([t.full_definition for t in themes], query=True)
    themes = [
        Theme(t.short_name, t.full_definition, v)
        for t, v in zip(themes, vectors, strict=True)
    ]
    names = [t.short_name for t in themes]
    df = add_theme_similarity_columns(df, themes)
    df = classify_by_max_theme(df, names, out_col="nearest_theme")
    df["relevance"] = df[names].max(axis=1).round(2)
    relevant = df["relevance"] >= THRESHOLD
    df["embedding_theme"] = df["nearest_theme"].where(relevant, OFF_TOPIC)

    out = df[["passage_id", "relevance", "nearest_theme", "embedding_theme", "text"]]
    out.to_csv(OUT / "02_coded.csv", index=False)
    print(f"{relevant.sum()} of {len(df)} passages at or above relevance {THRESHOLD}.")
    print(out["embedding_theme"].value_counts().to_string())
    print(f"Wrote {OUT / '02_coded.csv'}")


if __name__ == "__main__":
    main()
