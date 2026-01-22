from __future__ import annotations

from pathlib import Path

from src.llm_tasks import translate_to_english
from src.openai_client import get_client


def main() -> None:
    """Translate a Spanish transcript to English using LLM."""
    client = get_client()

    inp = Path("data/sample_transcripts/sample_spanish.md")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    outp = out_dir / "02_translated_english.md"

    spanish = inp.read_text(encoding="utf-8")
    english = translate_to_english(client, spanish)
    outp.write_text(english, encoding="utf-8")

    print("Wrote translation to:", outp)


if __name__ == "__main__":
    main()
