from __future__ import annotations

from pathlib import Path

from src.llm_tasks import extract_themes_for_question
from src.openai_client import get_client


def main() -> None:
    """Extract themes from transcript using LLM."""
    client = get_client()

    inp = Path("outputs/02_translated_english.md")
    if not inp.exists():
        inp = Path("data/sample_transcripts/sample_english.md")

    transcript = inp.read_text(encoding="utf-8")

    question = "How do facilitators experience integration into existing family services, and what helps or hinders it?"
    out_text = extract_themes_for_question(client, transcript, question)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "06_extracted_themes.txt"
    out_path.write_text(out_text, encoding="utf-8")

    print(f"Wrote: {out_path}\n")
    print(out_text)


if __name__ == "__main__":
    main()
