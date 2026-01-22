from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.coding import (
    add_theme_similarity_columns,
    classify_by_max_theme,
    embed_themes,
    load_themes,
)
from src.openai_client import get_client


def main() -> None:
    """Classify chunks by theme similarity using embeddings."""
    client = get_client()

    inp = Path("outputs/04_relevant_chunks.csv")
    if not inp.exists():
        raise FileNotFoundError(
            "Missing outputs/04_relevant_chunks.csv. Run step 04 first."
        )

    df = pd.read_csv(inp)
    df["embedding"] = df["embedding"].apply(json.loads)

    themes_path = Path("data/themes/help_themes.json")
    themes = load_themes(themes_path)
    themes = embed_themes(client, themes)

    df = add_theme_similarity_columns(df, themes)
    theme_cols = [t.short_name for t in themes]
    df = classify_by_max_theme(df, theme_cols, out_col="most_similar_theme")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "05_theme_classification.csv"

    df_to_save = df.copy()
    df_to_save["embedding"] = df_to_save["embedding"].apply(json.dumps)
    df_to_save.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print("\nTheme counts:")
    print(df["most_similar_theme"].value_counts())

    # Print top examples per theme
    print("\nTop examples per theme (top 2):")
    for t in theme_cols:
        top = (
            df[df["most_similar_theme"] == t].sort_values(by=t, ascending=False).head(2)
        )
        if len(top) == 0:
            continue
        print(f"\n== {t} ==")
        for _, row in top.iterrows():
            print(f"score={row[t]:.3f} | chunk_id={row['chunk_id']}")
            print(row["text"])


if __name__ == "__main__":
    main()
