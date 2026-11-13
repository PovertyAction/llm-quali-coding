"""Activity B: Create document chunks with embeddings."""

import os
from pathlib import Path

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def chunk_by_sections(
    markdown_text: str, source: str, chunk_size: int = 600, overlap: int = 100
) -> list[dict]:
    """Chunk markdown text by sections with overlap.

    Args:
        markdown_text: The markdown content
        source: Source document name
        chunk_size: Target tokens per chunk
        overlap: Overlap tokens between chunks

    Returns:
        List of chunk dictionaries

    """
    lines = markdown_text.split("\n")
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_section = "Introduction"
    chunk_id = 0

    for line in lines:
        # Detect section headers
        if line.startswith("#"):
            # Save previous chunk if it exists
            if current_chunk:
                chunks.append(
                    {
                        "chunk_id": f"{source}_chunk_{chunk_id}",
                        "project": source,
                        "section": current_section,
                        "text": "\n".join(current_chunk),
                        "token_count": current_tokens,
                    }
                )
                chunk_id += 1

                # Start new chunk with overlap
                overlap_lines = (
                    current_chunk[-5:] if len(current_chunk) > 5 else current_chunk
                )
                current_chunk = overlap_lines.copy()
                current_tokens = count_tokens("\n".join(current_chunk))

            # Update section title
            current_section = line.strip("#").strip()
            current_chunk = [line]
            current_tokens = count_tokens(line)
        else:
            # Add line to current chunk
            current_chunk.append(line)
            current_tokens = count_tokens("\n".join(current_chunk))

            # Split if chunk is too large
            if current_tokens > chunk_size:
                chunks.append(
                    {
                        "chunk_id": f"{source}_chunk_{chunk_id}",
                        "project": source,
                        "section": current_section,
                        "text": "\n".join(current_chunk),
                        "token_count": current_tokens,
                    }
                )
                chunk_id += 1

                # Start new chunk with overlap
                overlap_lines = current_chunk[-5:]
                current_chunk = overlap_lines.copy()
                current_tokens = count_tokens("\n".join(current_chunk))

    # Add final chunk
    if current_chunk:
        chunks.append(
            {
                "chunk_id": f"{source}_chunk_{chunk_id}",
                "project": source,
                "section": current_section,
                "text": "\n".join(current_chunk),
                "token_count": current_tokens,
            }
        )

    return chunks


def create_embeddings(chunks: list[dict]) -> list[dict]:
    """Add embeddings to chunks."""
    print("Creating embeddings...")

    # Filter out chunks with empty or whitespace-only text
    valid_chunks = [chunk for chunk in chunks if chunk.get("text", "").strip()]

    if len(valid_chunks) < len(chunks):
        print(f"  Filtered out {len(chunks) - len(valid_chunks)} empty chunks")

    if not valid_chunks:
        print("  No valid chunks to embed!")
        return chunks

    texts = [chunk["text"] for chunk in valid_chunks]

    # Create embeddings in batches
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  Processing batch {i // batch_size + 1} ({len(batch)} chunks)...")

        response = client.embeddings.create(input=batch, model="text-embedding-3-small")

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    # Add embeddings to valid chunks
    for chunk, embedding in zip(valid_chunks, all_embeddings):
        chunk["embedding"] = embedding

    print(f"✓ Created {len(all_embeddings)} embeddings")
    return chunks


def main() -> None:
    """Create document chunks with embeddings from markdown files."""
    markdown_dir = "outputs/markdown"
    output_file = "outputs/01_document_chunks.csv"

    print("Document Chunking and Embedding")
    print("=" * 60)

    # Load markdown files
    markdown_path = Path(markdown_dir)
    md_files = list(markdown_path.glob("*.md"))

    if not md_files:
        print(f"No markdown files found in {markdown_dir}")
        print("Please run 01_pdf_to_markdown.py first")
        return

    print(f"Found {len(md_files)} markdown files\n")

    all_chunks = []

    # Process each document
    for md_file in md_files:
        print(f"Processing: {md_file.name}")
        text = md_file.read_text(encoding="utf-8")
        source = md_file.stem

        chunks = chunk_by_sections(text, source)
        all_chunks.extend(chunks)

        print(f"  Created {len(chunks)} chunks")
        print(
            f"  Avg tokens: {sum(c['token_count'] for c in chunks) / len(chunks):.0f}"
        )

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Create embeddings
    all_chunks = create_embeddings(all_chunks)

    # Filter out chunks without embeddings before saving
    chunks_with_embeddings = [chunk for chunk in all_chunks if "embedding" in chunk]

    if len(chunks_with_embeddings) < len(all_chunks):
        print(f"  Saving only {len(chunks_with_embeddings)} chunks with embeddings")

    # Save to CSV
    df = pd.DataFrame(chunks_with_embeddings)
    df.to_csv(output_file, index=False)

    print(f"\n{'=' * 60}")
    print(f"Chunks saved to: {output_file}")
    print(f"Total chunks with embeddings: {len(chunks_with_embeddings)}")
    print(f"{'=' * 60}")

    # Show sample
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks_with_embeddings[:3], 1):
        print(f"\n{i}. {chunk['chunk_id']}")
        print(f"   Project: {chunk['project']}")
        print(f"   Section: {chunk['section']}")
        print(f"   Tokens: {chunk['token_count']}")
        print(f"   Text preview: {chunk['text'][:100]}...")


if __name__ == "__main__":
    main()
