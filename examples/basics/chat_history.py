"""Interactive chat interface with conversation history.

This script demonstrates:
- Maintaining conversation history across multiple turns
- How LLMs are stateless (you must send full history each time)
- Interactive chat loop with user input
- Building context over multiple exchanges

Type 'quit' or 'exit' to end the conversation.

Works with both OpenAI and Anthropic based on which API key is configured.
"""

from src.llm_client import create_completion, get_client, get_provider


def main():
    """Run an interactive chat session with conversation history."""
    # Get the provider and client (auto-detected from env vars)
    provider = get_provider()
    client = get_client()

    # Select model based on provider
    model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"

    print(f"\nUsing {provider} with model: {model}")

    # Initialize conversation with system message
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Be concise and friendly.",
        },
    ]

    print("✅ Chat session started!")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    # Chat loop
    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Check for exit commands
        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye!\n")
            break

        if not user_input:
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Get response from API (sending full conversation history)
        bot_response = create_completion(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            temperature=0.7,
        )

        # Add assistant message to history
        messages.append({"role": "assistant", "content": bot_response})

        # Print response
        print(f"\nAssistant: {bot_response}\n")


if __name__ == "__main__":
    main()
