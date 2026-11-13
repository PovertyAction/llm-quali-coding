# Session 02 — Embeddings & RAG

## Session goal

By the end of this session, you should be able to:

- explain what an **embedding** is and what it's used for,
- generate embeddings for texts,
- measure similarity (cosine / dot product) and retrieve *top-k* chunks,
- build a **mini-RAG**: *search first, then answer*.

> This session is **foundational**: what you learn applies to many use cases (internal chatbots, semantic search, assisted classification, etc.).

---

## Key concepts

### What is an embedding?

An embedding is a **numerical representation** of a text's meaning.

- It's a **vector** (list of numbers)
- Lives in a **vector space**
- Texts with similar meaning tend to be **close** in that space

**Important rule:** You can only compare embeddings if they were generated with the **same model** (same space).

**Example:**

```text
Text: "The survey was conducted in rural villages"
↓
Embedding: [0.023, -0.145, 0.892, ..., 0.334]
(1536 numbers with text-embedding-3-small)
```

Similar texts get similar vectors. Different meanings get distant vectors.

---

### Similarity: cosine similarity and dot product

The most common comparison is **cosine similarity** (intuition: "angle between vectors").

Many embedding models produce **normalized** vectors, and then:

- cosine similarity simplifies to a **dot product**
- this is efficient and scales well

**In practice:**

- Score close to 1 = very similar meaning
- Score close to 0 = unrelated
- Score close to -1 = opposite meaning (rare)

---

### What is RAG?

RAG = Retrieval-Augmented Generation

Instead of asking the LLM to "know" your documents, you:

1. **Retrieval**: Find relevant chunks (embeddings + search)
2. **Augmentation**: Add those chunks to the prompt
3. **Generation**: LLM responds using that context

Useful phrase:
> **Search first, then answer.**

**Key insight:** RAG is not fine-tuning. You're not changing the model—you're changing the **context**.

**Why RAG matters:**

- Works with private/proprietary data
- No need to retrain models
- Easy to update knowledge base
- Can cite sources
- Cost-effective at scale

---

## The RAG workflow

```text
Your documents
↓
Split into chunks
↓
Generate embeddings (one-time)
↓
Store embeddings
---
User question
↓
Generate question embedding
↓
Search similar chunks (retrieval)
↓
Build prompt with chunks (augmentation)
↓
LLM generates answer (generation)
```

---

## Guided activity

### Step 0 — Create and activate the environment (5 min)

Python relies heavily on external packages.
Even basic tasks often require importing libraries.

**In this project:**

- package versions are defined in the project configuration,
- a virtual environment isolates those packages from other projects.

**Run this command:**

If this is your first time setting up the project, run:

```bash
just get-started
```

If you've already set up the project before, just run:

```bash
just venv
```

### Preparation

Make sure you have:

- active environment (`.venv`)
- `.env` with `OPENAI_API_KEY` (required: every step creates embeddings with OpenAI's `text-embedding-3-small`)
  - `ANTHROPIC_API_KEY` is optional. If it is set too, Step C writes its answer with Claude
  - You can explicitly set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` in `.env` to choose
- the repo open in VS Code

---

### Step A — Generate embeddings (10 min)

Run:

```bash
python examples/embeddings/01_generate_embeddings.py
```

**What you should see:**

- Display of which LLM provider is being used (OpenAI or Anthropic)
- Confirmation that embeddings use OpenAI's model
- Print of the vector size (1536 dimensions for `text-embedding-3-small`)
- Preview of the first values
- Confirmation that the API works for embeddings

**What's happening:**

- Your text is being sent to OpenAI's embedding API
- The API returns a vector representation
- This vector captures the semantic meaning of your text

**Key observation:** Notice how different texts produce different vectors, but the vector length is always the same (1536 for `text-embedding-3-small`).

---

### Step B — Simple semantic search (15 min)

Run:

```bash
python examples/embeddings/02_similarity_search.py
```

**This script:**

1. Shows which LLM provider is configured (OpenAI or Anthropic)
2. Reads several short documents
3. Splits them into chunks (simple chunking)
4. Generates embeddings per chunk
5. Generates embedding of a question
6. Calculates similarities and shows the top-k

**What to observe:**

- The retrieved chunks should "make sense" with your question
- The top-1 and top-3 should be reasonable
- Notice the similarity scores (higher = more relevant)

**What's happening:**

- Each chunk gets converted to a vector
- Your question also becomes a vector
- We calculate how close each chunk vector is to the question vector
- We rank by similarity and return the top matches

**Key insight:** This is semantic search—we're finding meaning matches, not keyword matches.

---

### Step C — Mini-RAG (10–15 min)

Run:

```bash
python examples/embeddings/03_mini_rag.py
```

**This script does the same as Step B, but additionally:**

- Builds a prompt that includes the retrieved chunks
- Asks the LLM to respond using **only that context**
- Uses your configured LLM provider (OpenAI or Anthropic/Claude) for generation
- Prints the response

**What to observe:**

- Which LLM provider is being used for generation
- The response should reference the retrieved content
- If the question is not covered by the context, the model should say so (according to the instruction)
- Compare the answer to the chunks that were retrieved
- Notice that embeddings always use OpenAI, but the LLM generation can use either provider

**What's happening:**

```text
1. User asks: "What sampling method was used?"
2. System retrieves relevant chunks about sampling
3. System builds prompt: "Based on these documents: [chunks], answer: [question]"
4. LLM generates answer grounded in those chunks
```

**Key insight:** The LLM doesn't "know" your documents—you're giving it the relevant parts at query time.

---

## Mental model: RAG vs. other approaches

| Approach | When to use | Pros | Cons |
| ---------- | ------------- | ------ | ------ |
| **Simple LLM call** | General tasks, no specific knowledge needed | Simple, fast | Can't access your data |
| **Long context** | Small, static knowledge base | No retrieval needed | Expensive, may miss details |
| **RAG** | Large, changing knowledge base | Efficient, scalable, citable | Requires embedding setup |
| **Fine-tuning** | Need model to "speak" your domain | Model learns style/domain | Expensive, hard to update |

For most research use cases → RAG

---

## Quick exercises (extra)

1. **Change retrieval depth:** Modify `TOP_K` from 3 to 5 and compare results. Does more context help?

2. **Experiment with chunking:** Change the chunking strategy (e.g., by paragraphs vs. fixed size) and see how the top-k changes.

3. **Test edge cases:** Write a "difficult" question (little evidence in the docs) and observe how the mini-RAG behaves. Does it hallucinate or admit it doesn't know?

4. **Compare with keywords:** Try to find the same information using keyword search (Ctrl+F). Notice the difference with semantic search.

---

## Final activity — Build your own mini-RAG

Now that you've seen how the mini-RAG example works, it's time to build your own!

### Instructions

**Goal:** Create a mini-RAG system to answer questions about procurement procedures.

**Dataset:** Use the file located at `data/exercise/How to Create a Procurement Request.md`

**Your task:**

1. **Read and chunk** the procurement document
2. **Generate embeddings** for each chunk
3. **Implement the RAG workflow**:
4.
   - Accept user questions
   - Retrieve relevant chunks using similarity search
   - Build a prompt with the context
   - Generate answers using the LLM

5. **Test with questions** like:
6.
   - "What is the minimum advance time required for the delivery date according to the procurement request level in IPA's ProcessMaker system, and what happens if a user has not visited ProcessMaker recently?"
   - "If an IPA user does not find the Grant Allocation Code they need when creating a procurement request in ProcessMaker, who should they contact and what is the list of available codes based on?"

**Starting point:** Use `examples/embeddings/03_mini_rag.py` as your reference code and adapt it to work with the procurement document.

**You can:**

- Use Claude or GitHub Copilot Chat to help you write the code
- Ask questions about the implementation
- Experiment with different chunking strategies
- Adjust the number of retrieved chunks (`TOP_K`)

**Expected outcome:** A working script that can answer procurement-related questions based on the document content.

---

## Common issues and solutions

**"Embeddings are all similar (scores close to 1)"**
→ Your documents might be too similar, or chunks too small. Try more diverse content.

**"Top results don't make sense"**
→ Check your chunking strategy. Are chunks preserving semantic coherence?

**"LLM ignores the context"**
→ Make your prompt instruction clearer: "Answer ONLY using the provided context."

**"Too slow"**
→ Embeddings are generated once and stored. Only query embedding happens in real-time.

---

## What you've learned

- ✅ Embeddings convert text to searchable vectors
- ✅ Similarity search finds semantically related content
- ✅ RAG pattern: retrieve → augment → generate
- ✅ This works without fine-tuning or retraining models

---

## Bridge to next sessions

What comes next:

- **Session 03** (this afternoon): Apply this to qualitative coding workflows
- **Session 04** (this afternoon): Build a full knowledge chatbot with your internal documents

Future considerations:

- More robust chunking (tokens, overlap)
- Storing embeddings efficiently (files / vector DB)
- Traceability (citations, chunk IDs)
- Evaluation: Are we retrieving the right content? Does the response use the context correctly?

This session gives you the technical foundation to build these applications.

---

## Key takeaways

1. **Embeddings = semantic search**: Find by meaning, not keywords
2. **RAG = grounded generation**: LLM uses YOUR data, not just training knowledge
3. **Pattern is reusable**: Same approach works for many research applications
4. **No model training needed**: This is prompt engineering + smart retrieval

---

## Resources

- OpenAI Embeddings Guide: <https://platform.openai.com/docs/guides/embeddings>
- Anthropic Claude API: <https://docs.anthropic.com/en/api/getting-started>
- Understanding cosine similarity: <https://www.pinecone.io/learn/vector-similarity/>
- RAG best practices: <https://www.anthropic.com/index/contextual-retrieval>

---

**Ready for more?** Move on to Session 03 this afternoon to see embeddings in action for qualitative research.
