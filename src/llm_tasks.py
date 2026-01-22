from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .openai_client import load_config


def translate_to_english(client: OpenAI, spanish_text: str) -> str:
    """Translate Spanish text to English using LLM."""
    cfg = load_config()
    response = client.responses.create(
        model=cfg.llm_model,
        input=[
            {
                "role": "developer",
                "content": "You are a translator specializing in Spanish-to-English transcripts.",
            },
            {
                "role": "user",
                "content": "Translate the Spanish transcript below into English. Keep formatting as close as possible.\n\nTRANSCRIPT:\n"
                + spanish_text,
            },
        ],
    )
    return response.output_text


def extract_candidate_themes(
    client: OpenAI, english_transcript: str, research_question: str
) -> str:
    """Extract candidate themes from transcript based on research question."""
    cfg = load_config()
    response = client.responses.create(
        model=cfg.theme_extraction_model,
        reasoning={"effort": cfg.theme_extraction_reasoning_effort},
        input=[
            {
                "role": "developer",
                "content": (
                    "You are a PhD-level qualitative researcher. Your job is to propose a codebook (themes) from focus group transcripts. "
                    "Use rigorous, research-appropriate language."
                ),
            },
            {
                "role": "user",
                "content": (
                    "I will give you an English focus group transcript.\n"
                    "Please extract candidate themes specifically relevant to the research question below.\n"
                    "Return two sections: 'Helps integration' and 'Hinders integration'.\n\n"
                    f"RESEARCH QUESTION:\n{research_question}\n\n"
                    f"TRANSCRIPT:\n{english_transcript}"
                ),
            },
        ],
    )
    return response.output_text


def code_yes_no_for_theme(
    client: OpenAI, chunk_text: str, theme_definition: str
) -> str:
    """Return 'YES' or 'NO' depending on whether the chunk substantively relates to the theme."""
    cfg = load_config()
    response = client.responses.create(
        model=cfg.llm_model,
        reasoning={"effort": "low"},
        input=[
            {
                "role": "developer",
                "content": "You are a PhD qualitative researcher coding transcript chunks.",
            },
            {
                "role": "user",
                "content": (
                    "Decide whether the CHUNK below substantively discusses the THEME. "
                    "Only output one token: YES or NO.\n\n"
                    f"THEME:\n{theme_definition}\n\n"
                    f"CHUNK:\n{chunk_text}"
                ),
            },
        ],
    )
    return response.output_text.strip().split()[0].upper()


def code_nonverbal_cues(client: OpenAI, chunk_text: str) -> dict[str, Any]:
    """Extract non-verbal cue metadata from a chunk.

    Returns a dict with keys:
      - any_cues: 'YES'|'NO'
      - cue_type: short string (e.g. 'Laughter', 'Confusion', ...)

    The model is asked to return JSON only; we parse defensively.
    """
    cfg = load_config()
    response = client.responses.create(
        model=cfg.llm_model,
        reasoning={"effort": "low"},
        input=[
            {
                "role": "developer",
                "content": "You are a qualitative researcher extracting non-verbal cues from transcript notes.",
            },
            {
                "role": "user",
                "content": (
                    "From the CHUNK below, detect whether there is any explicit non-verbal cue info (e.g., laughter, pauses, confusion). "
                    'Return ONLY valid JSON with exactly these keys: {"any_cues": "YES"|"NO", "cue_type": <short string or empty>}.\n\n'
                    f"CHUNK:\n{chunk_text}"
                ),
            },
        ],
    )

    text = response.output_text.strip()
    try:
        data = json.loads(text)
        return {
            "any_cues": str(data.get("any_cues", "")).upper() or "NO",
            "cue_type": str(data.get("cue_type", "")).strip(),
        }
    except Exception:
        # Fallback: very simple heuristic
        lowered = chunk_text.lower()
        any_cues = (
            "YES"
            if any(
                k in lowered
                for k in ["laughter", "laugh", "(laughter)", "risas", "(risas)"]
            )
            else "NO"
        )
        cue_type = "Laughter" if any_cues == "YES" else ""
        return {"any_cues": any_cues, "cue_type": cue_type}
