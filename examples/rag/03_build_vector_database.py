"""Activity C: Build vector database and test retrieval."""

import ast
import os
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_chunks(csv_file: str) -> pd.DataFrame:
    """Load document chunks from CSV."""
    df = pd.read_csv(csv_file)

    # Filter out rows without embeddings (NaN values)
    df = df.dropna(subset=["embedding"])

    # Convert embedding strings back to lists
    # literal_eval only reads data; eval would run any code found in the file
    df["embedding"] = df["embedding"].apply(ast.literal_eval)

    return df


def embed_query(query: str) -> list[float]:
    """Create embedding for a query."""
    response = client.embeddings.create(input=query, model="text-embedding-3-small")
    return response.data[0].embedding


def retrieve_chunks(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Retrieve top-k most similar chunks for a query.

    Args:
        query: User question
        df: DataFrame with chunks and embeddings
        top_k: Number of chunks to retrieve

    Returns:
        DataFrame with top-k chunks and similarity scores

    """
    # Embed query
    query_embedding = embed_query(query)

    # Calculate similarities
    embeddings_matrix = np.array(df["embedding"].tolist())
    query_array = np.array(query_embedding).reshape(1, -1)

    similarities = cosine_similarity(query_array, embeddings_matrix)[0]

    # Add similarity scores to dataframe
    df_copy = df.copy()
    df_copy["similarity"] = similarities

    # Sort by similarity and return top-k
    top_chunks = df_copy.nlargest(top_k, "similarity")

    return top_chunks


def test_retrieval(df: pd.DataFrame) -> None:
    """Test retrieval with sample queries."""
    test_queries = [
        "What were the response rates in each project?",
        "What field challenges were encountered?",
        "What are best practices for community engagement?",
    ]

    print("\nTesting Retrieval")
    print("=" * 60)

    for query in test_queries:
        print(f'\nQuery: "{query}"')
        print("-" * 60)

        results = retrieve_chunks(query, df, top_k=5)

        for idx, (_, row) in enumerate(results.iterrows(), 1):
            print(
                f"\n{idx}. [Score: {row['similarity']:.3f}] {row['project']} > {row['section']}"
            )
            print(f"   {row['text'][:150]}...")

        print()


def save_vector_database(df: pd.DataFrame, output_file: str) -> None:
    """Save vector database to pickle file."""
    with open(output_file, "wb") as f:
        pickle.dump(df, f)
    print(f"Vector database saved to: {output_file}")


def main() -> None:
    """Build vector database from document chunks."""
    chunks_file = "outputs/01_document_chunks.csv"
    db_file = "outputs/02_vector_database.pkl"

    print("Vector Database Builder")
    print("=" * 60)

    # Load chunks
    print(f"Loading chunks from {chunks_file}...")
    df = load_chunks(chunks_file)
    print(f"Loaded {len(df)} chunks from {df['project'].nunique()} projects")

    # Show project distribution
    print("\nChunks per project:")
    print(df["project"].value_counts())

    # Test retrieval
    test_retrieval(df)

    # Save database
    save_vector_database(df, db_file)

    print("\n" + "=" * 60)
    print("Vector database ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
