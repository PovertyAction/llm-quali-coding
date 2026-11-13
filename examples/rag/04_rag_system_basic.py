"""Activity D: Implement basic RAG system."""

import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_vector_database(db_file: str) -> pd.DataFrame:
    """Load vector database from pickle file."""
    with open(db_file, "rb") as f:
        df = pickle.load(f)
    return df


def embed_query(query: str) -> list[float]:
    """Create embedding for a query."""
    response = client.embeddings.create(input=query, model="text-embedding-3-small")
    return response.data[0].embedding


def retrieve_chunks(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Retrieve top-k most similar chunks for a query."""
    query_embedding = embed_query(query)

    embeddings_matrix = np.array(df["embedding"].tolist())
    query_array = np.array(query_embedding).reshape(1, -1)

    similarities = cosine_similarity(query_array, embeddings_matrix)[0]

    df_copy = df.copy()
    df_copy["similarity"] = similarities

    top_chunks = df_copy.nlargest(top_k, "similarity")

    return top_chunks


def build_rag_prompt(query: str, chunks: pd.DataFrame) -> str:
    """Build prompt with retrieved context."""
    # Build context section
    context_parts = []
    for idx, (_, row) in enumerate(chunks.iterrows(), 1):
        context_parts.append(
            f"Document: {row['project']}, Section: {row['section']}\n"
            f"Content: {row['text']}\n"
        )

    context = "\n---\n".join(context_parts)

    # Build complete prompt
    prompt = f"""You are a research assistant helping analyze lessons learned from community health projects.

Context from project documentation:
---
{context}
---

User question: {query}

Instructions:
- Answer based ONLY on the provided context from project documents
- Cite specific projects and sections when making claims
- If a question cannot be answered from the context, say: "This information is not available in the provided project documents"
- Be specific and provide details from the documents
- Compare across projects when relevant

Answer:"""

    return prompt


def generate_answer(prompt: str) -> str:
    """Generate answer using LLM."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    return response.choices[0].message.content


def save_conversation(
    query: str, retrieved_chunks: pd.DataFrame, answer: str, output_file: str
) -> None:
    """Save conversation to JSONL file."""
    conversation = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_chunks": [
            {
                "project": row["project"],
                "section": row["section"],
                "similarity": float(row["similarity"]),
                "text_preview": row["text"][:200],
            }
            for _, row in retrieved_chunks.iterrows()
        ],
        "answer": answer,
    }

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(conversation, ensure_ascii=False) + "\n")


def rag_query(
    query: str, df: pd.DataFrame, top_k: int = 5, save_history: bool = True
) -> str:
    """Execute complete RAG pipeline.

    Args:
        query: User question
        df: Vector database
        top_k: Number of chunks to retrieve
        save_history: Whether to save conversation

    Returns:
        Generated answer

    """
    print(f"\nQuery: {query}")
    print("-" * 60)

    # Retrieve relevant chunks
    print(f"Retrieving top-{top_k} relevant chunks...")
    chunks = retrieve_chunks(query, df, top_k)

    avg_similarity = chunks["similarity"].mean()
    print(f"Average similarity: {avg_similarity:.3f}")

    # Show retrieved chunks
    print("\nRetrieved chunks:")
    for idx, (_, row) in enumerate(chunks.iterrows(), 1):
        print(f"  {idx}. [{row['similarity']:.3f}] {row['project']} > {row['section']}")

    # Build prompt
    print("\nGenerating answer...")
    prompt = build_rag_prompt(query, chunks)

    # Generate answer
    answer = generate_answer(prompt)

    # Save conversation
    if save_history:
        save_conversation(query, chunks, answer, "outputs/03_rag_conversations.jsonl")

    return answer


def main() -> None:
    """Run basic RAG system for querying project documentation."""
    db_file = "outputs/02_vector_database.pkl"

    print("RAG System - Basic Implementation")
    print("=" * 60)

    # Load database
    print(f"Loading vector database from {db_file}...")
    df = load_vector_database(db_file)
    print(f"Loaded {len(df)} chunks from {df['project'].nunique()} projects\n")

    # Example queries
    queries = [
        "What were the response rates in each project?",
        "What field challenges were encountered?",
        "What are best practices for community engagement?",
    ]

    print("Running example queries...\n")
    print("=" * 60)

    for query in queries:
        answer = rag_query(query, df, top_k=5)

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("\n" + "=" * 60 + "\n")

    # Interactive mode
    print("\nInteractive mode (type 'quit' to exit):")
    print("-" * 60)

    while True:
        user_query = input("\nYour question: ").strip()

        if user_query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_query:
            continue

        answer = rag_query(user_query, df, top_k=5)

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()
