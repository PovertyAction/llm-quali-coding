"""Activity E: RAG with metadata filtering."""

import os
import pickle
import re

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


def apply_filters(
    df: pd.DataFrame, project: str = None, section: str = None
) -> pd.DataFrame:
    """Apply metadata filters to chunks.

    Args:
        df: Full database
        project: Filter by project name (partial match)
        section: Filter by section name (partial match)

    Returns:
        Filtered dataframe

    """
    filtered = df.copy()

    if project:
        filtered = filtered[
            filtered["project"].str.contains(project, case=False, na=False)
        ]

    if section:
        filtered = filtered[
            filtered["section"].str.contains(section, case=False, na=False)
        ]

    return filtered


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
    context_parts = []
    for idx, (_, row) in enumerate(chunks.iterrows(), 1):
        context_parts.append(
            f"Document: {row['project']}, Section: {row['section']}\n"
            f"Content: {row['text']}\n"
        )

    context = "\n---\n".join(context_parts)

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


def rag_query_with_filters(
    query: str,
    df: pd.DataFrame,
    project_filter: str = None,
    section_filter: str = None,
    top_k: int = 5,
) -> str:
    """Execute RAG pipeline with metadata filters.

    Args:
        query: User question
        df: Vector database
        project_filter: Filter by project
        section_filter: Filter by section
        top_k: Number of chunks to retrieve

    Returns:
        Generated answer

    """
    print(f"\nQuery: {query}")

    # Apply filters
    if project_filter or section_filter:
        print("\nApplying filters:")
        if project_filter:
            print(f"  - Project contains: '{project_filter}'")
        if section_filter:
            print(f"  - Section contains: '{section_filter}'")

        original_count = len(df)
        df_filtered = apply_filters(df, project_filter, section_filter)
        print(f"  - Filtered chunks: {original_count} → {len(df_filtered)}")
    else:
        df_filtered = df

    if len(df_filtered) == 0:
        return "No chunks match the specified filters."

    # Retrieve relevant chunks
    print(f"\nRetrieving top-{top_k} relevant chunks...")
    chunks = retrieve_chunks(query, df_filtered, top_k)

    avg_similarity = chunks["similarity"].mean()
    print(f"Average similarity: {avg_similarity:.3f}")

    # Show retrieved chunks
    print("\nRetrieved chunks:")
    for idx, (_, row) in enumerate(chunks.iterrows(), 1):
        print(f"  {idx}. [{row['similarity']:.3f}] {row['project']} > {row['section']}")

    # Build prompt and generate answer
    print("\nGenerating answer...")
    prompt = build_rag_prompt(query, chunks)
    answer = generate_answer(prompt)

    return answer


def main() -> None:
    """Run RAG system with metadata filtering capabilities."""
    db_file = "outputs/02_vector_database.pkl"

    print("RAG System - With Metadata Filtering")
    print("=" * 60)

    # Load database
    print(f"Loading vector database from {db_file}...")
    df = load_vector_database(db_file)
    print(f"Loaded {len(df)} chunks from {df['project'].nunique()} projects\n")

    # Show available projects
    print("Available projects:")
    for project in df["project"].unique():
        count = len(df[df["project"] == project])
        print(f"  - {project}: {count} chunks")

    # Example queries with filters
    examples = [
        {
            "query": "What were the main findings?",
            "project": "Project_A",
            "section": None,
        },
        {
            "query": "What methodology was used?",
            "project": None,
            "section": "Methodology",
        },
        {
            "query": "What challenges were encountered?",
            "project": "Project_B",
            "section": "Challenges",
        },
    ]

    print("\n" + "=" * 60)
    print("Running example queries with filters...\n")

    for example in examples:
        print("=" * 60)
        answer = rag_query_with_filters(
            example["query"], df, example["project"], example["section"]
        )

        print("\n" + "-" * 60)
        print("ANSWER:")
        print("-" * 60)
        print(answer)
        print("\n")

    # Interactive mode
    print("=" * 60)
    print("Interactive mode (type 'quit' to exit)")
    print("Commands:")
    print("  - Normal query: Just type your question")
    print("  - Filter by project: [project:ProjectName] Your question")
    print("  - Filter by section: [section:SectionName] Your question")
    print("-" * 60)

    while True:
        user_input = input("\nYour question: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Parse filters from input: [project:Name] and [section:Name], in any
        # order and any letter case, then remove them from the question
        pattern = r"\[(project|section):([^\]]*)\]"
        found = {
            key.lower(): value.strip()
            for key, value in re.findall(pattern, user_input, flags=re.IGNORECASE)
        }
        project_filter = found.get("project")
        section_filter = found.get("section")
        query = re.sub(pattern, "", user_input, flags=re.IGNORECASE).strip()

        answer = rag_query_with_filters(
            example["query"], df, example["project"], example["section"]
        )

        print("\n" + "-" * 60)
        print("ANSWER:")
        print("-" * 60)
        print(answer)
        print("\n")

    # Interactive mode
    print("=" * 60)
    print("Interactive mode (type 'quit' to exit)")
    print("Commands:")
    print("  - Normal query: Just type your question")
    print("  - Filter by project: [project:ProjectName] Your question")
    print("  - Filter by section: [section:SectionName] Your question")
    print("-" * 60)

    while True:
        user_input = input("\nYour question: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Parse filters from input
        project_filter = None
        section_filter = None
        query = user_input

        # Check for project filter
        if "[project:" in user_input.lower():
            start = user_input.lower().index("[project:") + 9
            end = user_input.index("]", start)
            project_filter = user_input[start:end]
            query = user_input[: user_input.index("[project:")] + user_input[end + 1 :]

        # Check for section filter
        if "[section:" in user_input.lower():
            start = user_input.lower().index("[section:") + 9
            end = user_input.index("]", start)
            section_filter = user_input[start:end]
            query = user_input[: user_input.index("[section:")] + user_input[end + 1 :]

        query = query.strip()

        answer = rag_query_with_filters(query, df, project_filter, section_filter)

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()
