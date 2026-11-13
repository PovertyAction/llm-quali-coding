"""A simple script that demonstrates how to use the LLM API.

This script shows the basic pattern for making a chat completion request:
1. Get an authenticated client
2. Define your messages (system + user)
3. Call the API
4. Print the response

Works with both OpenAI and Anthropic based on which API key is configured.
"""

from src.llm_client import create_completion, get_client, get_provider


def main():
    """Send a simple chat message and print the response."""
    # Get the provider and client (auto-detected from env vars)
    provider = get_provider()
    client = get_client()

    # Select model based on provider
    model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"

    print(f"\nUsing {provider} with model: {model}\n")

    # Create a chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like in Bogota today?"},
    ]

    response_text = create_completion(
        client=client,
        provider=provider,
        model=model,
        messages=messages,
        temperature=0.7,
    )

    # Print the response
    print("✅ Chat completion successful!\n")
    print("Assistant response:")
    print(response_text)


if __name__ == "__main__":
    main()
