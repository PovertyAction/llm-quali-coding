from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from .chunking import Chunk
from .embeddings import get_embedding
from .similarity import dot_similarity


@dataclass
class Theme:
    """A codebook theme: name, definition, embedding, and optional coding guidance."""

    short_name: str
    full_definition: str  # "Name: definition"
    embedding: list[float] | None = None
    include: str = ""  # when to apply the theme
    exclude: str = ""  # when not to apply it
    examples: str = ""  # example quotes


def load_markdown(path: Path) -> str:
    """Load markdown file content as string."""
    return path.read_text(encoding="utf-8")


def themes_from_data(data: list) -> list[Theme]:
    """Build themes from a codebook list.

    Each item is either a "Name: definition" string or a dict with the keys
    name, definition and, optionally, include, exclude, examples.
    """
    themes: list[Theme] = []
    for item in data:
        if isinstance(item, str):
            short = item.split(":")[0].strip() if ":" in item else item[:40].strip()
            themes.append(Theme(short_name=short, full_definition=item))
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        definition = str(item.get("definition") or "").strip()
        themes.append(
            Theme(
                short_name=name,
                full_definition=f"{name}: {definition}" if definition else name,
                include=str(item.get("include") or "").strip(),
                exclude=str(item.get("exclude") or "").strip(),
                examples=str(item.get("examples") or "").strip(),
            )
        )
    return themes


def themes_to_data(themes: list[Theme]) -> list[dict[str, str]]:
    """Turn themes into dicts for the codebook editor and for saving."""
    return [
        {
            "name": t.short_name,
            "definition": t.full_definition.removeprefix(t.short_name)
            .removeprefix(":")
            .strip(),
            "include": t.include,
            "exclude": t.exclude,
            "examples": t.examples,
        }
        for t in themes
    ]


def load_themes(path: Path) -> list[Theme]:
    """Load themes from JSON file and parse into Theme objects."""
    return themes_from_data(json.loads(path.read_text(encoding="utf-8")))


def dump_themes(themes: list[Theme]) -> str:
    """Codebook as JSON; plain "Name: definition" strings when nothing else is set."""
    if any(t.include or t.exclude or t.examples for t in themes):
        data: list = themes_to_data(themes)
    else:
        data = [t.full_definition for t in themes]
    return json.dumps(data, ensure_ascii=False, indent=2)


def save_themes(path: Path, themes: list[Theme]) -> None:
    """Save a codebook as JSON, readable by load_themes."""
    path.write_text(dump_themes(themes), encoding="utf-8")


def theme_prompt_text(theme: Theme) -> str:
    """Render a theme with its coding guidance, for prompts and for embedding."""
    lines = [theme.full_definition]
    if theme.include:
        lines.append(f"Include when: {theme.include}")
    if theme.exclude:
        lines.append(f"Exclude when: {theme.exclude}")
    if theme.examples:
        lines.append(f"Examples: {theme.examples}")
    return "\n".join(lines)


def build_chunk_dataframe(chunks: list[Chunk]) -> pd.DataFrame:
    """Convert list of Chunk objects to pandas DataFrame."""
    return pd.DataFrame([{"chunk_id": c.chunk_id, "text": c.text} for c in chunks])


def embed_chunks(
    client: OpenAI, df: pd.DataFrame, text_col: str = "text"
) -> pd.DataFrame:
    """Generate embeddings for text chunks in DataFrame."""
    embeddings: list[list[float]] = []
    for text in tqdm(df[text_col].tolist(), desc="Embedding chunks"):
        embeddings.append(get_embedding(client, text))
    df = df.copy()
    df["embedding"] = embeddings
    return df


def compute_relevance_scores(
    df: pd.DataFrame, question_embedding: list[float]
) -> pd.DataFrame:
    """Compute similarity scores between chunks and a question embedding."""
    df = df.copy()
    df["question_similarity"] = [
        dot_similarity(e, question_embedding) for e in df["embedding"]
    ]
    return df


def filter_relevant(df: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    """Filter DataFrame to keep only chunks above relevance threshold."""
    return (
        df[df["question_similarity"] >= threshold]
        .sort_values("question_similarity", ascending=False)
        .reset_index(drop=True)
    )


def embed_themes(client: OpenAI, themes: list[Theme]) -> list[Theme]:
    """Generate embeddings for theme definitions."""
    out: list[Theme] = []
    for t in tqdm(themes, desc="Embedding themes"):
        emb = get_embedding(client, t.full_definition)
        out.append(
            Theme(
                short_name=t.short_name,
                full_definition=t.full_definition,
                embedding=emb,
            )
        )
    return out


def add_theme_similarity_columns(df: pd.DataFrame, themes: list[Theme]) -> pd.DataFrame:
    """Add similarity score columns for each theme to DataFrame."""
    df = df.copy()
    for t in themes:
        assert t.embedding is not None
        df[t.short_name] = [dot_similarity(e, t.embedding) for e in df["embedding"]]
    return df


def classify_by_max_theme(
    df: pd.DataFrame, theme_columns: list[str], out_col: str = "most_similar_theme"
) -> pd.DataFrame:
    """Classify chunks by the theme with highest similarity score."""
    df = df.copy()
    df[out_col] = df[theme_columns].idxmax(axis=1)
    return df
