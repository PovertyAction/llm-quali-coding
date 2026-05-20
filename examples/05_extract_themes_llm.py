from __future__ import annotations

from pathlib import Path

from src.llm_tasks import extract_general_themes
from src.openai_client import get_client, get_transcript_path, load_config


def main() -> None:
    """Extract general themes from transcript using LLM (inductive coding)."""
    client = get_client()
    cfg = load_config()

    inp = get_transcript_path(cfg.transcript_language)
    if not inp.exists():
        raise FileNotFoundError(
            f"Transcript file not found for language '{cfg.transcript_language}': {inp}\n"
            "Set TRANSCRIPT_LANGUAGE in your .env file and ensure the corresponding file exists."
        )

    print(
        f"Reading transcript from: {inp}  (TRANSCRIPT_LANGUAGE={cfg.transcript_language})"
    )
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
