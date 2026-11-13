from __future__ import annotations

from pathlib import Path

from src.chunking import chunk_text_simple
from src.embeddings import embed_text, embed_texts
from src.llm_client import get_provider
from src.rag import answer_with_context
from src.similarity import dot_similarity

TOP_K = 3
DOCS_DIR = Path("data/sample_docs")


def read_docs() -> list[tuple[str, str]]:
    """Read all markdown documents from the sample docs directory."""
    docs = []
    for fp in sorted(DOCS_DIR.glob("*.md")):
        docs.append((fp.stem, fp.read_text(encoding="utf-8")))
    if not docs:
        raise FileNotFoundError(f"No .md files found in {DOCS_DIR}")
    return docs


def main() -> None:
    """Demonstrate a mini RAG pipeline with retrieval and generation."""
    provider = get_provider()
    print(f"Using LLM provider: {provider}\n")

    docs = read_docs()

    # 1) chunking
    chunks = []
    for doc_id, text in docs:
        chunks.extend(chunk_text_simple(text, doc_id=doc_id, max_chars=600, overlap=80))

    # 2) chunk embeddings
    chunk_texts = [c.text for c in chunks]
    chunk_embs = embed_texts(chunk_texts)

    # 3) question
    question = "What practices reduced data quality errors in the field?"
    q_emb = embed_text(question)

    # 4) retrieval top-k
    scored = []
    for c, e in zip(chunks, chunk_embs):
        scored.append((dot_similarity(q_emb, e), c))
    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [c.text for _, c in scored[:TOP_K]]

    print("\nQuestion:")
    print(question)

    print("\nRetrieved context (top-k):\n")
    for i, txt in enumerate(top_chunks, start=1):
        preview = txt.replace("\n", " ")
        preview = (preview[:240] + "...") if len(preview) > 240 else preview
        print(f"[{i}] {preview}")

    # 5) generation using context
    answer = answer_with_context(question, top_chunks)

    print("\nAnswer (mini-RAG):\n")
    print(answer)


if __name__ == "__main__":
    main()
