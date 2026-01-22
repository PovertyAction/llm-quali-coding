from __future__ import annotations

import json
import re
from pathlib import Path

from src.chunking import make_chunks
from src.coding import build_chunk_dataframe, embed_chunks
from src.embeddings import get_embedding
from src.openai_client import get_client


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
        speaker_match = re.match(r"^([A-ZÁÉÍÓÚÑ\s]+\d*):\s*(.*)$", line)

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


def group_responses_by_moderator(responses: list[dict]) -> list[dict]:
    """Group facilitator responses under each moderator question.

    Args:
        responses: List of speaker-text dicts

    Returns:
        List of grouped responses with 'moderator_question', 'responses', and 'joint' keys

    """
    grouped_responses = []
    current_moderator_turn = None
    facilitator_responses = []

    for response in responses:
        speaker = response["speaker"].upper()

        # Check if this is a moderator
        if "MODERADOR" in speaker or speaker == "MODERATOR":
            # If there's a current moderator turn and collected responses, add them
            if current_moderator_turn is not None:
                # Join facilitator responses into a single string
                combined_facilitator_text = "\n".join(
                    [f"{r['speaker']}: {r['text']}" for r in facilitator_responses]
                )

                # Create joint text (moderator question + responses)
                joint_text = f"{current_moderator_turn}\n\n{combined_facilitator_text}"

                grouped_responses.append(
                    {
                        "moderator_question": current_moderator_turn,
                        "responses": combined_facilitator_text,
                        "joint": joint_text,
                    }
                )

            # Start a new moderator turn
            current_moderator_turn = response["text"]
            facilitator_responses = []
        else:
            # Collect facilitator/participant responses
            facilitator_responses.append(response)

    # Add the last moderator turn and collected responses if any
    if current_moderator_turn is not None and facilitator_responses:
        combined_facilitator_text = "\n".join(
            [f"{r['speaker']}: {r['text']}" for r in facilitator_responses]
        )
        joint_text = f"{current_moderator_turn}\n\n{combined_facilitator_text}"

        grouped_responses.append(
            {
                "moderator_question": current_moderator_turn,
                "responses": combined_facilitator_text,
                "joint": joint_text,
            }
        )

    return grouped_responses


def chunk_by_moderator_question(text: str) -> str:
    """Chunk transcript by moderator questions with participant responses.

    Each chunk contains a moderator question followed by all participant responses.

    Args:
        text: Full transcript text with speaker labels

    Returns:
        Combined text where each section is separated by '---'

    """
    # Step 1: Parse speakers
    speakers = parse_speakers(text)

    # Step 2: Group by moderator questions
    grouped = group_responses_by_moderator(speakers)

    # Step 3: Extract joint text and combine
    chunks = [item["joint"] for item in grouped if item["joint"].strip()]

    return "\n\n---\n\n".join(chunks)


def main() -> None:
    """Create embeddings for transcript chunks and save to CSV."""
    client = get_client()

    # Use the Spanish sample transcript (or translated English if available)
    default_inp = Path("data/sample_transcripts/sample_spanish.md")
    translated = Path("outputs/01_translated_english.md")
    inp = translated if translated.exists() else default_inp

    text = inp.read_text(encoding="utf-8")

    # Chunk by moderator questions (each chunk = moderator question + participant responses)
    chunked_text = chunk_by_moderator_question(text)
    print("Chunked transcript by moderator questions")

    # Use make_chunks to further split if needed (based on min_chars)
    chunks = make_chunks(chunked_text, min_chars=250)
    df = build_chunk_dataframe(chunks)

    df = embed_chunks(client, df, text_col="text")

    # Save embeddings as JSON strings (keeps this repo lightweight and dependency-free)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "01_chunks_with_embeddings.csv"

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
