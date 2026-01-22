# LLM Qualitative Coding

This repository is a walk through an end-to-end **qualitative coding workflow** using LLMs:

1. Local setup + first API call
2. Translation (optional)
3. Embeddings
4. Relevance filtering (question ↔ chunks)
5. Theme classification (themes ↔ chunks)
6. Optional: theme extraction with an LLM
7. Optional: direct coding with an LLM
8. Optional: coding non-verbal cues with structured output
9. Optional: exploratory clustering (inductive coding)

This repo is designed for **learning foundations**. You can swap in your own transcripts and codebook later.

---

## Quickstart (after cloning)

### 1) Create a virtual environment and install dependencies

1. This repo uses **uv** (recommended) and optionally **just**.

   If you have `just` installed:

   ```bash
   just get-started
   ```

   If you do not have `just`:

   ```bash
   uv sync
   uv pip install -e .
   ```

2. Activate the virtual environment: `.venv/Scripts/activate.ps1`
  
### 2) Add your API key

1. Copy `.env.example` → `.env`:

   ```bash
   cp .env.example .env
   ```

2. Open `.env` and set:

```text
OPENAI_API_KEY=YOUR_KEY_HERE
```

> Your `.env` file is ignored by Git (see `.gitignore`). Do not commit it.

---

### 3) Run the scripts (recommended order)

#### Step 1 — Test connection

```bash
python examples/01_test_connection.py
```

#### Step 2 — Translate transcript (optional)

```bash
python examples/02_translate_transcript.py
```

#### Step 3 — Create embeddings

```bash
uv run python examples/03_create_embeddings.py
```

#### Step 4 — Relevance filtering (question ↔ chunks)

```bash
python examples/04_relevance_filtering.py
```

#### Step 5 — Theme classification (themes ↔ chunks)

```bash
python examples/05_theme_classification_embeddings.py
```

---

## What you'll find in this repo

- `docs/` — session notes and explanations
- `data/sample_transcripts/` — small example transcript files
- `data/themes/` — an example theme list (codebook-style)
- `src/` — reusable functions (client, embeddings, similarity, chunking, coding)
- `examples/` — runnable scripts in pedagogical order
- `outputs/` — created automatically when you run examples (ignored by Git)

---

## How the pipeline works (mental model)

```text
Transcript(s)
  ↓
Chunking (split into paragraphs / segments)
  ↓
Embeddings for each chunk
  ↓
Question embedding
  ↓
Relevance score (dot product) → keep relevant chunks
  ↓
Theme embeddings (your codebook)
  ↓
Theme similarity scores
  ↓
Classification (argmax or threshold)
```

---

## Customizing for your own project

1. Replace the sample transcript in `data/sample_transcripts/`
2. Replace the theme list in `data/themes/help_themes.json`
3. Update the research question in `examples/04_relevance_filtering.py`
4. Re-run scripts 03 → 05

---

## Notes on cost and privacy

- Embeddings are usually cheap and fast.
- LLM calls for theme extraction and direct coding can be more expensive.
- Never upload sensitive transcripts unless you have approval and appropriate safeguards.

---

## License

MIT (suggested). Add a LICENSE file if you want to publish.
