# LLM Qualitative Coding

Training materials for learning to use Large Language Models for qualitative research coding and analysis.

## What is this?

A hands-on training repository that teaches:

- How to make API calls to LLMs from Python
- What embeddings are and how to use them for semantic search
- How to apply LLM capabilities to qualitative coding workflows
- Practical techniques: theme extraction, direct coding, inductive clustering

## Who is this for?

Research staff who want to:

- Use LLMs programmatically for qualitative analysis
- Work with interview transcripts and qualitative data
- Build AI-augmented coding and analysis tools
- Understand embeddings and semantic similarity

No prior LLM experience required. Basic Python knowledge helpful.

## Repository structure

```text
llm-quali-coding/
├── docs/                    # Session guides with step-by-step instructions
│   ├── session_01_setup.md
│   ├── session_02_embeddings_rag.md
│   └── session_03_quali_coding.md
├── examples/                # Runnable Python scripts
│   ├── 01_test_connection.py
│   ├── 02_translate_transcript.py
│   ├── 03_create_embeddings.py
│   ├── 04_relevance_filtering.py
│   ├── 05_theme_classification_embeddings.py
│   ├── 06_extract_themes_llm.py
│   ├── 07_direct_coding_llm.py
│   ├── 08_nonverbal_coding_llm.py
│   └── 09_inductive_clustering.py
├── src/                     # Reusable code modules
│   ├── chunking.py
│   ├── coding.py
│   ├── embeddings.py
│   ├── llm_tasks.py
│   ├── openai_client.py
│   └── similarity.py
└── data/                    # Sample data for exercises
    ├── sample_transcripts/
    └── themes/
```

## Training sessions

### Day 1: Foundations

1. **Session 01** (1 hour): Local setup and your first LLM API call
2. **Session 02** (1 hour): Introduction to embeddings and RAG

### Day 2: Qualitative Coding Applications

1. **Session 03** (2 hours): Qualitative coding techniques with LLMs

## Getting started

1. Clone this repository
2. Follow the setup instructions in `docs/session_01_setup.md`
3. Complete sessions in order

Each session builds on the previous one.

## Requirements

- Python 3.11+
- OpenAI API key
- Code editor (VS Code or Positron recommended)
- Basic command line familiarity

## Support

- Session guides contain detailed instructions and troubleshooting
- Example scripts include inline comments
- Common issues documented in each session guide

## Learning outcomes

By the end of this training, you will:

- Have a working local LLM development environment
- Understand how to use LLM APIs programmatically
- Know what embeddings are and when to use them for qualitative analysis
- Be able to apply multiple LLM-based coding techniques to qualitative data
- Have practical experience with theme extraction, classification, and clustering

---

**Start here:** `docs/session_01_setup.md`
