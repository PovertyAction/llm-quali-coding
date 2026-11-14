"""Step 2: find relevant passages and code them with local embeddings.

Uses an open-weights embedding model on the laptop. A passage's relevance is
its similarity to the closest codebook theme. Passages below the threshold are
coded "Fuera de tema"; the rest get the most similar theme.

Writes outputs/local/02_coded.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.coding import load_themes
from src.pipeline import code_with_embeddings, make_embedder

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
    themes = load_themes(Path("data/fictitious/codebook_ficticio.json"))
    df = code_with_embeddings(
        pd.read_csv(passages), themes, make_embedder("local"), THRESHOLD, OFF_TOPIC
    )
    relevant = df["relevance"] >= THRESHOLD

    out = df[["passage_id", "relevance", "nearest_theme", "embedding_theme", "text"]]
    out.to_csv(OUT / "02_coded.csv", index=False)
    print(f"{relevant.sum()} of {len(df)} passages at or above relevance {THRESHOLD}.")
    print(out["embedding_theme"].value_counts().to_string())
    print(f"Wrote {OUT / '02_coded.csv'}")


if __name__ == "__main__":
    main()
