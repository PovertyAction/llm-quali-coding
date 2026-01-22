# Session 03 — Qualitative Coding with Embeddings (and optional LLM coding)

This session builds on Sessions 01–02 and shows how to use embeddings to support **qualitative coding**.

> Important: this is a teaching workflow. In real projects, you should agree on analytic choices with your PI / research team.

---

## Big picture

We will take a transcript and:

1. split it into chunks
2. create embeddings for each chunk
3. compute **relevance** to a research question
4. (optional) prune irrelevant chunks
5. compare relevant chunks to a **pre-specified theme list** (codebook)
6. assign each chunk to a theme (classification)

Optionally, we will:

- extract candidate themes with an LLM (codebook draft)
- do direct LLM coding (YES/NO for a single theme)
- extract structured non-verbal cue codes
- explore inductive clusters with KMeans

---

## Key idea 1: Embeddings live in a vector space

Embeddings are **numerical representations of semantic meaning**.
They are vectors, so embeddings must exist in the same vector space to be compared meaningfully.

Once everything is embedded, the workflow is straightforward:

- embed chunks
- embed a question and/or themes
- compare vectors

---

## Key idea 2: Similarity is usually cosine (often dot product)

For most embedding systems, cosine similarity is used to measure how close two meanings are.

Many embedding models normalize vectors (roughly to the same length). When vectors are normalized, cosine similarity simplifies to a **dot product**, which is faster.

---

## Part A — Relevance filtering (question ↔ chunks)

### Why we do this

Transcripts are wide-ranging. If you only care about one research question, a first useful step is to keep chunks that are likely relevant.

### Steps

1. Create an embedding for each chunk.
2. Create an embedding for the research question.
3. Compute a relevance score for each chunk (dot product).
4. Sort and optionally prune based on a threshold.

In this repo, see:

- `examples/03_create_embeddings.py`
- `examples/04_relevance_filtering.py`

---

## Part B — Theme scoring (themes ↔ chunks)

### What are themes here?

Themes are a short text definitions for codes you care about.
They can come from:

- a PI / theory-driven codebook
- implementation partner priorities
- prior studies
- or an exploratory LLM pass (optional)

### Steps

1. Define your theme list (codebook-style text definitions).
2. Create an embedding per theme.
3. For each chunk, compute similarity to each theme.

In this repo, see:

- `data/themes/help_themes.json`
- `examples/05_theme_classification_embeddings.py`

---

## Part C — Classification (choose your rule)

Once you have theme similarity scores, you must decide how to assign labels.
Common options:

- **Single-label**: assign the theme with the maximum score (`argmax`)
- **Multi-label**: assign all themes above a threshold
- **Continuous**: use similarity scores directly (e.g., as a covariate)
- **Exploratory**: cluster embeddings to discover latent groupings

This repo demonstrates **single-label argmax** for simplicity.

---

## Optional appendix — When you might use an LLM coder

Embeddings are often reproducible and cheap, but sometimes:

- themes are extremely subtle
- you want explicit decisions (YES/NO)
- you want to code meta information (tone, laughter, confusion)

Then you can use an LLM as a coder with a strict instruction.

See:

- `examples/06_extract_themes_llm.py`
- `examples/07_direct_coding_llm.py`
- `examples/08_nonverbal_coding_llm.py`

---

## Optional appendix — Inductive coding with clustering

Clustering is exploratory. It can help you see groupings you did not anticipate.

This repo includes an optional script:

- `examples/09_inductive_clustering.py`

It:

- builds a matrix of chunk embeddings
- runs KMeans
- prints a few example chunks per cluster

---

## Suggested discussion prompts

- Which chunks were filtered out as irrelevant? Are we okay with that?
- Do the top chunks per theme look right?
- Are there themes that overlap too much?
- Would multi-label classification fit better?
- If we used an LLM coder, how would we evaluate reliability?
