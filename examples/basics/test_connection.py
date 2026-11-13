"""Test that your local environment can successfully connect to the LLM API.

This script tests connectivity with either OpenAI or Anthropic based on
which API key is configured. If this script runs without errors and prints
a short response, your setup is correct.
"""

from src.llm_client import create_completion, get_client, get_provider


def main():
    """Test the LLM API connection."""
    # Get the provider and client (auto-detected from env vars)
    provider = get_provider()
    client = get_client()

    # Select model based on provider
    model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"

    print(f"\n🔌 Testing connection to {provider}...")
    print(f"Using model: {model}\n")

    # Simple test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]

    # Test the connection
    response_text = create_completion(
        client=client, provider=provider, model=model, messages=messages
    )

    print("✅ Connection successful!\n")
    print("Model response:")
    print(response_text)


if __name__ == "__main__":
    main()
