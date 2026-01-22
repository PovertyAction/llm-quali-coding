from __future__ import annotations

from src.openai_client import get_client, load_config


def main() -> None:
    """Test OpenAI API connection with a simple prompt."""
    cfg = load_config()
    client = get_client()

    prompt = [
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]

    response = client.responses.create(model=cfg.llm_model, input=prompt)
    print("Connection OK ✅")
    print(f"Model: {cfg.llm_model}")
    print("Response:", response.output_text)


if __name__ == "__main__":
    main()
