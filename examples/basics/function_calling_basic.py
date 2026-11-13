"""Demonstrates function calling (without execution).

This script shows how to:
- Declare a function schema (lookup_weather)
- Have the model decide when to call the function
- Parse tool call responses from the model

This example shows the decision-making but doesn't execute the function.

Works with both OpenAI and Anthropic based on which API key is configured.
"""

import json

from src.llm_client import (
    create_completion_with_tools,
    extract_tool_calls,
    get_client,
    get_provider,
)


def main():
    """Demonstrate function calling without execution."""
    # Get the provider and client (auto-detected from env vars)
    provider = get_provider()
    client = get_client()

    # Select model based on provider
    model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"

    print(f"\nUsing {provider} with model: {model}\n")

    # Define the function schema (tools) in OpenAI format
    # The adapter will convert to Anthropic format if needed
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Lookup the weather for a city or zip code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city_name": {
                            "type": "string",
                            "description": "The city name",
                        },
                        "zip_code": {
                            "type": "string",
                            "description": "The zip code",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }
    ]

    # Make API call with tools parameter
    response = create_completion_with_tools(
        client=client,
        provider=provider,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful weather assistant.",
            },
            {
                "role": "user",
                "content": "What's the temperature in Celsius in Bogota?",
            },
        ],
        tools=tools,
        tool_choice="auto",
    )

    # Check if the model chose to call a function
    print("✅ Function calling demonstration:\n")

    tool_calls = extract_tool_calls(response, provider)

    if tool_calls:
        # Model decided to call a function
        for tool_call in tool_calls:
            function_name = tool_call["name"]
            arguments = tool_call["arguments"]

            # For OpenAI, arguments come as JSON string
            # For Anthropic, arguments come as dict
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            print(f"Model chose to call: {function_name}")
            print(f"Arguments: {arguments}")
    else:
        # Model responded with text instead
        print("Model responded with text instead of calling a function:")
        if provider == "openai":
            print(response.choices[0].message.content)
        else:
            print(response.content[0].text)


if __name__ == "__main__":
    main()
