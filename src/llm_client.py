"""LLM client factory and adapter functions supporting both OpenAI and Anthropic.

This module provides a unified interface for working with multiple LLM providers
while keeping the API differences visible for educational purposes.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def get_provider():
    """Detect and return which provider to use.

    Detection order:
    1. Check LLM_PROVIDER environment variable ("openai" or "anthropic")
    2. Auto-detect based on available API keys:
       - Only ANTHROPIC_API_KEY → use Anthropic
       - Only OPENAI_API_KEY → use OpenAI
       - Both keys present → use Anthropic (default)
       - Neither key → raise error

    Returns:
        str: Either "openai" or "anthropic"

    Raises:
        ValueError: If no API keys are found or invalid provider specified

    """
    # Check for explicit provider selection
    explicit_provider = os.getenv("LLM_PROVIDER", "").lower()
    if explicit_provider:
        if explicit_provider not in ["openai", "anthropic"]:
            raise ValueError(
                f"Invalid LLM_PROVIDER: {explicit_provider}. "
                "Must be 'openai' or 'anthropic'"
            )
        return explicit_provider

    # Auto-detect based on available keys
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    if not has_anthropic and not has_openai:
        raise ValueError(
            "No API keys found. Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "in your .env file.\n"
            "Get keys from:\n"
            "- OpenAI: https://platform.openai.com/api-keys\n"
            "- Anthropic: https://console.anthropic.com/settings/keys"
        )

    # If both keys present, default to Anthropic
    if has_anthropic:
        return "anthropic"

    return "openai"


def get_client(provider=None):
    """Get an authenticated LLM client.

    Args:
        provider: Optional provider override ("openai" or "anthropic").
                 If not specified, uses get_provider() to auto-detect.

    Returns:
        OpenAI or Anthropic: An authenticated client instance

    Raises:
        ValueError: If provider is invalid or API key is missing

    """
    if provider is None:
        provider = get_provider()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Get one from: "
                "https://platform.openai.com/api-keys"
            )
        return OpenAI(api_key=api_key)

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Get one from: "
                "https://console.anthropic.com/settings/keys"
            )
        return Anthropic(api_key=api_key)

    raise ValueError(f"Invalid provider: {provider}")


def _extract_system_message(messages):
    """Extract system message from messages array for Anthropic.

    Anthropic uses a separate 'system' parameter instead of including
    system messages in the messages array.

    Args:
        messages: List of message dicts with "role" and "content"

    Returns:
        tuple: (system_content, filtered_messages)
               system_content is None if no system message found

    """
    system_content = None
    filtered_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            filtered_messages.append(msg)

    return system_content, filtered_messages


def _convert_tools_to_anthropic(tools):
    """Convert OpenAI tool format to Anthropic format.

    OpenAI format:
        {"type": "function", "function": {"name": "...", "parameters": {...}}}

    Anthropic format:
        {"name": "...", "description": "...", "input_schema": {...}}

    Args:
        tools: List of tool dicts in OpenAI format

    Returns:
        list: Tools in Anthropic format

    """
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            anthropic_tools.append(
                {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                }
            )
    return anthropic_tools


def create_completion(client, provider, model, messages, **kwargs):
    """Create a chat completion with provider-specific handling.

    Args:
        client: Authenticated client (OpenAI or Anthropic instance)
        provider: Provider name ("openai" or "anthropic")
        model: Model name (provider-specific)
        messages: List of message dicts with "role" and "content"
        **kwargs: Additional parameters (temperature, etc.)

    Returns:
        str: The response text content

    """
    if provider == "openai":
        response = client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content

    if provider == "anthropic":
        # Extract system message
        system_content, filtered_messages = _extract_system_message(messages)

        # Build request parameters
        request_params = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }

        if system_content:
            request_params["system"] = system_content

        response = client.messages.create(**request_params)
        return response.content[0].text

    raise ValueError(f"Invalid provider: {provider}")


def create_streaming_completion(client, provider, model, messages, **kwargs):
    """Create a streaming chat completion with provider-specific handling.

    Yields text chunks uniformly regardless of provider.

    Args:
        client: Authenticated client (OpenAI or Anthropic instance)
        provider: Provider name ("openai" or "anthropic")
        model: Model name (provider-specific)
        messages: List of message dicts with "role" and "content"
        **kwargs: Additional parameters (temperature, etc.)

    Yields:
        str: Text chunks as they arrive

    """
    if provider == "openai":
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True, **kwargs
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    elif provider == "anthropic":
        # Extract system message
        system_content, filtered_messages = _extract_system_message(messages)

        # Build request parameters
        request_params = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }

        if system_content:
            request_params["system"] = system_content

        with client.messages.stream(**request_params) as stream:
            yield from stream.text_stream

    else:
        raise ValueError(f"Invalid provider: {provider}")


def create_completion_with_tools(client, provider, model, messages, tools, **kwargs):
    """Create a chat completion with function calling support.

    Args:
        client: Authenticated client (OpenAI or Anthropic instance)
        provider: Provider name ("openai" or "anthropic")
        model: Model name (provider-specific)
        messages: List of message dicts with "role" and "content"
        tools: List of tool/function definitions (OpenAI format)
        **kwargs: Additional parameters (temperature, tool_choice, etc.)

    Returns:
        object: Provider-specific response object with tool calls

    """
    if provider == "openai":
        return client.chat.completions.create(
            model=model, messages=messages, tools=tools, **kwargs
        )

    if provider == "anthropic":
        # Extract system message
        system_content, filtered_messages = _extract_system_message(messages)

        # Convert tools to Anthropic format
        anthropic_tools = _convert_tools_to_anthropic(tools)

        # Build request parameters
        request_params = {
            "model": model,
            "messages": filtered_messages,
            "tools": anthropic_tools,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }

        if system_content:
            request_params["system"] = system_content

        # Remove tool_choice if present (different format in Anthropic)
        request_params.pop("tool_choice", None)

        return client.messages.create(**request_params)

    raise ValueError(f"Invalid provider: {provider}")


def extract_tool_calls(response, provider):
    """Extract tool call information from response.

    Args:
        response: Provider-specific response object
        provider: Provider name ("openai" or "anthropic")

    Returns:
        list: List of dicts with "name" and "arguments" keys
              Returns empty list if no tool calls

    """
    if provider == "openai":
        if response.choices[0].message.tool_calls:
            tool_calls = []
            for tool_call in response.choices[0].message.tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )
            return tool_calls
        return []

    if provider == "anthropic":
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({"name": block.name, "arguments": block.input})
        return tool_calls

    raise ValueError(f"Invalid provider: {provider}")
