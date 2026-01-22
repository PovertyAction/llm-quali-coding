# Session 02 — Embeddings + RAG (Foundations)

This session explains **embeddings** and the **RAG** pattern (Retrieval-Augmented Generation).

## Goals

By the end of this session, you should be able to:

- explain what an embedding is and why we use them
- create embeddings for text chunks
- compute similarity scores (cosine / dot product)
- retrieve the most relevant chunks for a question (top-k)
- explain how this retrieval step becomes RAG when you pass retrieved text into an LLM prompt

---

## 1) What is an embedding?

Embeddings are **numerical representations of semantic meaning**.

- an embedding is a **vector** (a list of numbers)
- vectors must be in the **same vector space** to compare them meaningfully
- once you have embeddings, you can do simple, powerful math to compare text

---

## 2) Similarity: cosine vs dot product

For most embedding systems, the **cosine similarity** between two vectors measures semantic similarity.

Many embedding models normalize vectors (roughly between -1 and 1). When vectors are normalized:

> cosine similarity simplifies to a **dot product**

This is computationally efficient, and we will use it.

---

## 3) Creating embeddings with the API

We will write a helper function that:

1. initializes a client
2. calls the embeddings endpoint
3. returns the embedding vector

In this repo, that logic lives in:

- `src/openai_client.py` (client)
- `src/embeddings.py` (embeddings call)

Run:

```bash
uv run python examples/03_create_embeddings.py
```

---

## 4) Retrieval (semantic search)

Once we embed transcript chunks, we can embed a **question** and find the most relevant chunks.

- embed question
- compute similarity(question, chunk)
- sort
- keep top-k

Run:

```bash
uv run python examples/04_relevance_filtering.py
```

---

## 5) From retrieval to RAG

RAG is a simple pattern:

1. **Retrieve** relevant chunks (using embeddings)
2. **Generate** an answer using an LLM, with those chunks included as context

In this repo, `examples/05_theme_classification_embeddings.py` focuses on *coding* rather than answering.
A pure RAG chatbot is a separate use case, but the retrieval step is identical.

---

## What to remember

- embeddings turn meaning into numbers
- similarity scores give you relevance
- retrieval is the foundation of RAG
