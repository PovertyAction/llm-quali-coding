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
