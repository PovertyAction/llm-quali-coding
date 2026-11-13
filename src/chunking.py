from __future__ import annotations

import re
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


def parse_speakers(text: str) -> list[dict]:
    """Parse transcript into speaker-text pairs.

    Args:
        text: Full transcript text with speaker labels

    Returns:
        List of dicts with 'speaker' and 'text' keys

    """
    lines = text.split("\n")
    speakers = []
    current_speaker = None
    current_text = []

    for line in lines:
        # Match speaker labels (e.g., "MODERADOR:", "FACILITADOR 1:")
        speaker_match = re.match(r"^([A-ZÁÉÍÓÚÑ\s]+\d*):\s*(.*)$", line, re.IGNORECASE)

        if speaker_match:
            # Save previous speaker's text
            if current_speaker and current_text:
                speakers.append(
                    {"speaker": current_speaker, "text": " ".join(current_text).strip()}
                )

            # Start new speaker
            current_speaker = speaker_match.group(1).strip()
            current_text = (
                [speaker_match.group(2)] if speaker_match.group(2).strip() else []
            )
        else:
            # Continue current speaker's text
            if line.strip():
                current_text.append(line.strip())

    # Add last speaker
    if current_speaker and current_text:
        speakers.append(
            {"speaker": current_speaker, "text": " ".join(current_text).strip()}
        )

    return speakers


@dataclass
class DocChunk:
    """Represents a chunk of text from a document.

    Attributes:
        doc_id: Identifier of the source document
        chunk_id: Sequential identifier for this chunk within the document
        text: The text content of the chunk

    """

    doc_id: str
    chunk_id: int
    text: str


def chunk_text_simple(
    text: str, doc_id: str, max_chars: int = 700, overlap: int = 100
) -> list[DocChunk]:
    """Chunk text using simple character-based chunking.

    - max_chars: chunk size in characters
    - overlap: overlap between consecutive chunks

    This is intentionally simple for Session 02.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[DocChunk] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(DocChunk(doc_id=doc_id, chunk_id=chunk_id, text=chunk))
            chunk_id += 1
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks
