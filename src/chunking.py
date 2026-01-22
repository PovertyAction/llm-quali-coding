from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with an ID and content."""

    chunk_id: int
    text: str


def split_markdown_into_paragraphs(md_text: str) -> list[str]:
    """Split markdown text into paragraphs separated by blank lines.

    This is intentionally simple for teaching purposes.
    """
    lines = md_text.replace("\r\n", "\n").split("\n")

    paras: list[str] = []
    buf: list[str] = []
    for line in lines:
        if line.strip() == "":
            if buf:
                para = "\n".join(buf).strip()
                if para:
                    paras.append(para)
                buf = []
            continue
        # Drop markdown headings (optional)
        if line.strip().startswith("#") and not buf:
            continue
        buf.append(line)

    if buf:
        para = "\n".join(buf).strip()
        if para:
            paras.append(para)

    return paras


def merge_short_paragraphs(
    paragraphs: Iterable[str], min_chars: int = 300
) -> list[str]:
    """Merge consecutive short paragraphs to create more stable chunks."""
    merged: list[str] = []
    buf = ""
    for p in paragraphs:
        buf = p if not buf else buf + "\n\n" + p
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


def make_chunks(md_text: str, min_chars: int = 300) -> list[Chunk]:
    """Create numbered chunks from markdown text."""
    paras = split_markdown_into_paragraphs(md_text)
    merged = merge_short_paragraphs(paras, min_chars=min_chars)
    return [Chunk(chunk_id=i + 1, text=t) for i, t in enumerate(merged)]
