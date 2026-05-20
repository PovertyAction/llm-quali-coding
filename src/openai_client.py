from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Maps known language codes to their bundled transcript files.
# For any other language, files are resolved as data/sample_transcripts/sample_<lang>.md
_TRANSCRIPT_FILES: dict[str, Path] = {
    "en": Path("data/sample_transcripts/sample_english.md"),
    "es": Path("data/sample_transcripts/sample_spanish.md"),
}


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for OpenAI models and transcript language."""

    llm_model: str
    theme_extraction_model: str
    theme_extraction_reasoning_effort: str
    embedding_model: str
    transcript_language: str


def load_config() -> ModelConfig:
    """Load environment variables and return model configuration.

    Expected variables:
      - OPENAI_API_KEY
      - LLM_MODEL (default: gpt-5-mini)
      - THEME_EXTRACTION_MODEL (default: gpt-5)
      - THEME_EXTRACTION_REASONING_EFFORT (default: high)
      - EMBEDDING_MODEL (default: text-embedding-3-large)
      - TRANSCRIPT_LANGUAGE (default: en) — language code for transcript content
    """
    load_dotenv()

    return ModelConfig(
        llm_model=os.getenv("LLM_MODEL", "gpt-5-mini"),
        theme_extraction_model=os.getenv("THEME_EXTRACTION_MODEL", "gpt-5"),
        theme_extraction_reasoning_effort=os.getenv(
            "THEME_EXTRACTION_REASONING_EFFORT", "high"
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        transcript_language=os.getenv("TRANSCRIPT_LANGUAGE", "en").strip().lower(),
    )


def get_transcript_path(lang: str) -> Path:
    """Return the transcript file path for the given language code.

    Known languages (en, es) map to their bundled files.
    Any other code falls back to data/sample_transcripts/sample_<lang>.md.
    """
    return _TRANSCRIPT_FILES.get(
        lang, Path(f"data/sample_transcripts/sample_{lang}.md")
    )


def get_client() -> OpenAI:
    """Create an OpenAI client using OPENAI_API_KEY from environment."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create a .env file (see .env.example) and set OPENAI_API_KEY."
        )
    return OpenAI(api_key=api_key)
