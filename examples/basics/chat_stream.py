"""Demonstrates streaming responses from the LLM API.

This script demonstrates streaming responses, which:
- Show tokens appearing progressively (like ChatGPT interface)
- Provide better user experience for long responses
- Use the same API call with stream=True parameter

Works with both OpenAI and Anthropic based on which API key is configured.
"""

from src.llm_client import (
    create_streaming_completion,
    get_client,
    get_provider,
)


def main():
    """Send a chat message and stream the response as it's generated."""
    # Get the provider and client (auto-detected from env vars)
    provider = get_provider()
    client = get_client()

    # Select model based on provider
    model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"

    print(f"\nUsing {provider} with model: {model}\n")

    # Create messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": ("Write a short description of the poverty probability index."),
        },
    ]

    # Stream and print the response as it arrives
    print("✅ Streaming response:\n")
    for text_chunk in create_streaming_completion(
        client=client,
        provider=provider,
        model=model,
        messages=messages,
        temperature=0.7,
    ):
        print(text_chunk, end="", flush=True)

    print("\n")  # Add newline at the end


if __name__ == "__main__":
    main()
