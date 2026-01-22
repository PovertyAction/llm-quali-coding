from __future__ import annotations

from pathlib import Path

from src.llm_tasks import extract_general_themes
from src.openai_client import get_client


def main() -> None:
    """Extract general themes from transcript using LLM (inductive coding)."""
    client = get_client()

    # Try to load translated version first, fall back to Spanish original
    inp = Path("outputs/02_translated_english.md")
    if not inp.exists():
        inp = Path("data/sample_transcripts/sample_spanish.md")
        if not inp.exists():
            raise FileNotFoundError(
                "No transcript found. Run 01_translate_transcript.py first or ensure "
                "data/sample_transcripts/sample_spanish.md exists."
            )

    print(f"Reading transcript from: {inp}")
    transcript = inp.read_text(encoding="utf-8")

    print("\nExtracting themes from transcript (this may take a moment)...\n")
    out_text = extract_general_themes(client, transcript)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "04_extracted_themes.txt"
    out_path.write_text(out_text, encoding="utf-8")

    print(f"✅ Wrote: {out_path}\n")
    print("=" * 60)
    print("EXTRACTED THEMES")
    print("=" * 60)
    print(out_text)


if __name__ == "__main__":
    main()
