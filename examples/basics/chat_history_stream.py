"""Interactive chat with conversation history and streaming responses.

This script combines:
- Conversation history management (remembering past messages)
- Streaming responses (progressive token display)
- Interactive chat interface

This is the most production-like example, closest to how real chatbots work.

Type 'quit' or 'exit' to end the conversation.

Works with both OpenAI and Anthropic based on which API key is configured.
"""

from src.llm_client import (
    create_streaming_completion,
    get_client,
    get_provider,
)


def main():
    """Run an interactive chat with conversation history and streaming."""
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

    print("✅ Chat session started (streaming mode)!")
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

        # Stream and collect the response
        print("\nAssistant: ", end="", flush=True)
        bot_response = ""
        for text_chunk in create_streaming_completion(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            temperature=0.7,
        ):
            print(text_chunk, end="", flush=True)
            bot_response += text_chunk

        print("\n")  # Add newline after streaming completes

        # Add assistant message to history
        messages.append({"role": "assistant", "content": bot_response})


if __name__ == "__main__":
    main()
