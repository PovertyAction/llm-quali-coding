"""Translate the IPA Best Bets document from English to Spanish.

This script:
- Reads the IPA Best Bets document in English
- Translates it to Spanish using the LLM API
- Saves the Spanish version to the data folder

Works with both OpenAI and Anthropic based on which API key is configured.
"""

from pathlib import Path

from src.llm_client import create_completion, get_client, get_provider


def translate_to_spanish(text: str, provider: str, client, model: str) -> str:
    """Translate English text to Spanish using the LLM API.

    Args:
        text: The English text to translate
        provider: The LLM provider ("openai" or "anthropic")
        client: The authenticated client
        model: The model to use for translation

    Returns:
        The translated Spanish text

    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translator specializing in "
                "academic and policy documents. "
                "Translate the following document from English to Spanish. "
                "Maintain the markdown formatting exactly as it appears. "
                "Use formal, professional Spanish appropriate for "
                "policy and research documents. "
                "Preserve all technical terms and proper nouns appropriately."
            ),
        },
        {
            "role": "user",
            "content": f"Translate this document to Spanish:\n\n{text}",
        },
    ]

    response_text = create_completion(
        client=client,
        provider=provider,
        model=model,
        messages=messages,
        temperature=0.3,  # Lower temperature for more consistent translation
    )

    return response_text


def main():
    """Translate the IPA Best Bets document and save to data folder."""
    # Get the provider and client (auto-detected from env vars)
    provider = get_provider()
    client = get_client()

    # Select quality model based on provider
    model = "gpt-4o" if provider == "openai" else "claude-haiku-4-5"

    print(f"\nUsing {provider} with model: {model}")

    # Define file paths
    input_file = Path("data/ipa-best-bets-2025.md")
    output_file = Path("data/ipa-best-bets-2025-es.md")

    print(f"\nReading {input_file}...")

    # Read the English document
    with open(input_file, encoding="utf-8") as f:
        english_text = f.read()

    print(f"✅ Document loaded ({len(english_text)} characters)")
    print("\nTranslating to Spanish (this may take a moment)...")

    # Translate to Spanish
    spanish_text = translate_to_spanish(english_text, provider, client, model)

    print(f"✅ Translation complete ({len(spanish_text)} characters)")

    # Create data directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the Spanish translation
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(spanish_text)

    print(f"\nSpanish version saved to: {output_file}")
    print("\nTranslation complete!\n")


if __name__ == "__main__":
    main()
