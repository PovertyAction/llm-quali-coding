from __future__ import annotations

from src.llm_client import create_completion, get_client, get_provider


def answer_with_context(
    question: str,
    context_chunks: list[str],
    model: str | None = None,
) -> str:
    """Ask the LLM to answer using provided context.

    We instruct the model to rely on context and to say when
    information is missing.

    Args:
        question: User's question
        context_chunks: List of relevant context chunks
        model: Optional model name override. If not provided, uses
            provider-specific default.

    Returns:
        str: Model's answer based on the provided context

    """
    provider = get_provider()
    client = get_client()

    # Set provider-specific default models
    if model is None:
        model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"

    context = "\n\n---\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente útil. Responde la pregunta usando SOLO "
                "el contexto provisto. Si el contexto no contiene la "
                "respuesta, dilo explícitamente. Mantén la respuesta concisa."
            ),
        },
        {
            "role": "user",
            "content": f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{context}",
        },
    ]

    return create_completion(
        client=client,
        provider=provider,
        model=model,
        messages=messages,
        max_tokens=1024,
    )
