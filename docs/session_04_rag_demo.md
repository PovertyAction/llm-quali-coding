# Session 04 — RAG: Retrieval Augmented Generation for projects

## Session goal

By the end of this session, you should be able to:

- convert **PDFs to markdown** for text processing,
- create **embeddings** from project documentation,
- build a **vector database** for semantic search,
- implement a **RAG (Retrieval Augmented Generation)** system,
- query project knowledge (response rates, field practices, challenges),
- understand when RAG improves LLM responses vs. direct prompting.

> This session builds on Sessions 02 and 03, applying embeddings to project documentation and institutional knowledge management.
>
> **Important:** This is a teaching workflow. In real projects, consider data privacy and document sensitivity when building knowledge bases.

---

## Key concepts

### What is RAG?

**RAG (Retrieval Augmented Generation)** combines two steps:

1. **Retrieval**: Find relevant documents from your knowledge base
2. **Generation**: LLM generates answers using retrieved context

**Why RAG?**

- LLMs don't know your specific project details
- Training custom models is expensive and time-consuming
- RAG gives LLMs access to your documents without retraining
- Maintains up-to-date information as documents change
- Provides source citations for answers

**Key insight:** RAG turns your project documents into a queryable knowledge base that the LLM can reference when answering questions.

---

### The RAG workflow

```text
Project PDFs (3 reports)
↓
Convert PDFs → Markdown
↓
Chunk documents (by section, paragraph, or semantic unit)
↓
Generate embeddings for each chunk
↓
Store in vector database (CSV or vector DB)
↓
User asks a question
↓
Embed the question
↓
Retrieve top-k most similar chunks
↓
Send question + retrieved chunks to LLM
↓
LLM generates answer with context
↓
Return answer with source citations
```

---

### PDF to markdown conversion

**Problem:** PDFs are not easily processed by LLMs. They contain formatting, images, tables, and metadata that complicates text extraction.

**Solution:** Convert PDFs to clean markdown text that preserves structure (headers, lists, paragraphs) but removes unnecessary formatting.

**Benefits:**

- Clean, parseable text for chunking
- Preserves document structure (sections, subsections)
- Easier to process programmatically
- Better quality embeddings

**Tools:** We'll use `docling`, which reads the page layout to find headings and tables and writes them as markdown. It includes its own OCR for scanned pages, so there is nothing else to install.

---

### Chunking strategies for project documents

**Goal:** Break documents into meaningful units that can be embedded and retrieved independently.

**Common strategies:**

| Strategy | Best for | Chunk size |
|----------|----------|------------|
| **By section** | Structured reports with clear headers | Variable (100-1000 tokens) |
| **Fixed size** | Homogeneous documents | 500-1000 tokens with overlap |
| **Semantic units** | Mixed content (paragraphs, lists, tables) | Variable, context-aware |
| **By metadata** | Multi-document corpora | Depends on document structure |

**For project reports, we recommend:**

- **Section-based chunking**: Each major section becomes a chunk
- **Preserve context**: Include section headers in chunks
- **Optimal size**: 300-800 tokens per chunk (roughly 1-3 paragraphs)
- **Add overlap**: Include 50-100 tokens from previous chunk for continuity

---

### Retrieval: Finding relevant chunks

**How it works:**

1. User asks: "What were the main field challenges in Project A?"
2. System embeds the question → vector
3. Compares question vector to all chunk vectors (cosine similarity)
4. Returns top-k most similar chunks (typically k=3-5)
5. These chunks provide context to the LLM

**Retrieval strategies:**

| Strategy | When to use |
|----------|-------------|
| **Top-k retrieval** | Most common, simple and effective |
| **Threshold-based** | When you want minimum relevance score |
| **Hybrid (keyword + semantic)** | For technical documents with specific terms |
| **Metadata filtering** | When querying specific projects or sections |

**Key insight:** Good retrieval is critical—if the right chunks aren't retrieved, the LLM can't generate accurate answers.

---

### Generation: LLM with context

**Prompt structure:**

```text
You are a research assistant helping analyze project documentation.

Context from project documents:
---
[Chunk 1: Most relevant section...]
[Chunk 2: Next relevant section...]
[Chunk 3: Additional context...]
---

User question: What were the main field challenges in Project A?

Instructions:
- Answer based ONLY on the provided context
- Cite specific documents and sections
- If information is not in context, say so
- Be specific and concise
```

**Why this works:**

- LLM has project-specific information in context
- Grounded responses (not hallucinated)
- Can cite sources
- Respects document content

---

### When to use RAG vs. direct prompting

| Approach | Best for | Pros | Cons |
|----------|----------|------|------|
| **Direct prompting** | General questions, known facts | Simple, fast, cheap | No project-specific knowledge |
| **RAG** | Project-specific queries | Accurate, cited, updatable | Requires setup, slower |
| **Fine-tuning** | Repeated patterns, specialized tasks | Best performance | Expensive, static knowledge |

**Use RAG when:**

- Querying project-specific information (field practices, response rates, lessons learned)
- You need source citations
- Documents update frequently
- You want to combine knowledge from multiple projects

**Use direct prompting when:**

- Asking general research methods questions
- Brainstorming or ideation
- No document reference needed

---

## Guided activities

### Preparation

Make sure you have:

- Create environment (`just venv`)
- `.env` with `OPENAI_API_KEY`
- The repo open in VS Code
- Completed Sessions 02 and 03 (embeddings and qualitative coding)
- Three project PDF reports in `data/project_reports/`
- Activate environment: `.venv/Scripts/activate.ps1`

All required packages (docling, openai, pandas, numpy, scikit-learn, etc.) will be installed automatically when you run `just venv`.

---

### Activity A — Convert PDFs to markdown

Run:

```bash
python examples/rag/01_pdf_to_markdown.py
```

**What this script does:**

1. Scans `data/project_reports/` for PDF files
2. For each PDF:
   - Extracts text using `docling` (the first run downloads its models, about 500 MB, and takes a few minutes)
   - Preserves markdown structure (headers, lists, paragraphs)
   - Cleans formatting artifacts
   - Saves to `outputs/markdown/project_name.md`
3. Generates a summary report showing:
   - Number of PDFs processed
   - Document sizes
   - Section structure detected

**What to observe:**

- How many sections were detected per document?
- Is the markdown structure clean and readable?
- Are tables and lists preserved correctly?
- Are there any extraction errors or artifacts?

**Common issues:**

- **Scanned PDFs (images)**: docling runs OCR on them, but expect more errors than with text PDFs—read the output
- **Complex tables**: May not convert perfectly—inspect output
- **Multi-column layouts**: Text order might be wrong—verify manually

**What's happening:**

- The script intelligently identifies document structure
- Headers (H1, H2, H3) are detected from font size and formatting
- Paragraphs are preserved as text blocks
- Lists and bullet points are maintained
- This creates clean, LLM-ready text

**Key insight:** Good markdown extraction is the foundation of quality RAG. Garbage in, garbage out—spend time validating your converted documents.

---

### Activity B — Create document chunks with embeddings

Run:

```bash
python examples/rag/02_create_document_chunks.py
```

**What this script does:**

1. Loads all markdown files from `outputs/markdown/`
2. **Parses document structure**: Identifies sections, subsections, paragraphs
3. **Chunks by semantic units**:
   - Each major section becomes a chunk
   - Includes section header for context
   - Target size: 300-800 tokens
   - Overlaps 100 tokens between chunks for continuity
4. **Adds metadata** to each chunk:
   - Source document (project name)
   - Section title
   - Page/location reference
   - Chunk index
5. Generates embeddings for each chunk
6. Saves to `outputs/07_document_chunks.csv`

**Chunking strategy:**

```text
Project_A.md
├─ Introduction (200 tokens) → Chunk 1 [metadata: Project_A, Introduction]
├─ Methodology
│  ├─ Sampling Design (450 tokens) → Chunk 2 [metadata: Project_A, Methodology > Sampling]
│  └─ Data Collection (600 tokens) → Chunk 3 [metadata: Project_A, Methodology > Data Collection]
├─ Findings
│  ├─ Response Rates (350 tokens) → Chunk 4 [metadata: Project_A, Findings > Response Rates]
│  └─ Field Challenges (700 tokens) → Chunk 5 [metadata: Project_A, Findings > Challenges]
└─ Conclusions (400 tokens) → Chunk 6 [metadata: Project_A, Conclusions]
```

**What to observe:**

- How many chunks were created per document?
- What's the average chunk size (tokens)?
- Are chunks too small or too large?
- Does metadata look correct (project names, sections)?
- Are embeddings generated for all chunks?

**Output CSV structure:**

```text
chunk_id | project | section | text | embedding | token_count
---------|---------|---------|------|-----------|------------
chunk_1  | Project_A | Introduction | [text...] | [0.12, -0.45, ...] | 215
chunk_2  | Project_A | Methodology > Sampling | [text...] | [0.34, 0.21, ...] | 430
...
```

**What's happening:**

- Each chunk becomes a searchable unit in your knowledge base
- Embeddings capture semantic meaning of each section
- Metadata enables filtering (e.g., "only search Project A")
- This creates a vector database of project knowledge

**Key insight:** Chunking decisions matter! Chunks that are too small lack context; chunks that are too large dilute relevance. Section-based chunking balances both.

---

### Activity C — Build vector database and test retrieval

Run:

```bash
python examples/rag/03_build_vector_database.py
```

**What this script does:**

1. Loads document chunks with embeddings from Activity B
2. **Indexes chunks** for fast similarity search:
   - Converts embeddings to numpy arrays
   - Builds cosine similarity search index
   - Stores metadata for filtering
3. **Tests retrieval** with sample queries:
   - "What were the response rates in each project?"
   - "What field challenges were encountered?"
   - "What are best practices for community engagement?"
4. For each query:
   - Embeds the question
   - Retrieves top-5 most similar chunks
   - Shows similarity scores
   - Displays source metadata (project, section)
5. Saves indexed database to `outputs/08_vector_database.pkl`

**Sample output:**

```text
Query: "What were the response rates in each project?"

Top 5 retrieved chunks:
1. [Score: 0.87] Project_A > Findings > Response Rates
   "The final response rate was 68%, exceeding our target of 65%..."
   
2. [Score: 0.82] Project_B > Results > Participation Metrics
   "We achieved a 72% completion rate among enrolled participants..."
   
3. [Score: 0.79] Project_C > Data Quality > Coverage Analysis
   "Initial response rate was 55%, with follow-up increasing to 63%..."
   
4. [Score: 0.71] Project_A > Methodology > Sampling Strategy
   "We anticipated a 60-70% response rate based on pilot testing..."
   
5. [Score: 0.68] Project_B > Appendix > Statistical Power
   "Sample size calculations assumed a 70% response rate..."
```

**What to observe:**

- Are the top-retrieved chunks actually relevant to the query?
- What are the similarity scores? (>0.7 is good, >0.8 is excellent)
- Do chunks from multiple projects get retrieved?
- Are there irrelevant chunks in the top-5?
- Does retrieval capture both direct answers and related context?

**What's happening:**

- The system is doing semantic search—finding meaning matches, not keyword matches
- "Response rates" query retrieves chunks about participation, completion, coverage—all semantically related
- Metadata shows which project and section each chunk comes from
- This is the **retrieval** part of RAG

**Key insight:** Test retrieval independently before adding the LLM. If retrieval fails, the entire RAG system fails.

**Try this:** Modify the test queries and see what gets retrieved. Try specific vs. general questions.

---

### Activity D — Implement RAG system (basic)

Run:

```bash
python examples/rag/04_rag_system_basic.py
```

**What this script does:**

1. Loads the vector database from Activity C
2. **Interactive query interface**:
   - Prompts user for a question
   - Retrieves top-k relevant chunks (k=5 by default)
   - Constructs a prompt with retrieved context
   - Sends to LLM for answer generation
   - Returns answer with source citations
3. **Implements RAG pipeline**:
   - Query → Embed → Retrieve → Generate → Cite
4. Saves conversation history to `outputs/09_rag_conversations.jsonl`

**RAG prompt template:**

```text
You are a research assistant helping analyze lessons learned from three community health projects.

Context from project documentation:
---
Document: Project_A, Section: Findings > Response Rates
Content: [retrieved chunk 1...]

Document: Project_B, Section: Methodology > Field Practices
Content: [retrieved chunk 2...]

Document: Project_C, Section: Challenges > Community Engagement
Content: [retrieved chunk 3...]
---

User question: {user_question}

Instructions:
- Answer based ONLY on the provided context from project documents
- Cite specific projects and sections when making claims
- If a question cannot be answered from the context, say: "This information is not available in the provided project documents"
- Be specific and provide details from the documents
- Compare across projects when relevant
```

**Example interaction:**

```text
You: What were the main challenges with community engagement?

RAG System:
[Retrieving relevant chunks from 3 projects...]
[Retrieved 5 chunks with avg similarity: 0.81]

Based on the project documentation:

**Project A** faced challenges with trust-building in initial outreach. 
The report notes: "Community members were initially skeptical of external 
researchers, requiring 3-4 preliminary meetings before consent" (Project_A > 
Findings > Challenges).

**Project B** encountered language barriers. They report: "Lack of materials 
in indigenous languages limited participation among 30% of target population" 
(Project_B > Lessons Learned > Accessibility).

**Project C** identified timing issues: "Scheduling conflicts with agricultural 
seasons resulted in lower turnout at community meetings" (Project_C > Field 
Notes > Scheduling).

All three projects recommend: early community partnership, culturally 
appropriate materials, and flexible scheduling.
```

**What to observe:**

- Does the answer stay grounded in the documents?
- Are citations specific and accurate?
- Does the system compare across projects when asked?
- What happens when you ask about information not in the documents?
- Is the answer quality better than without RAG?

**What's happening:**

- **Retrieval**: Finds chunks most semantically similar to your question
- **Augmentation**: Adds these chunks as context in the prompt
- **Generation**: LLM reads context and generates informed answer
- **Citation**: LLM references specific projects/sections (if instructed)

**Key insight:** RAG gives LLMs "working memory" of your documents. The LLM doesn't memorize your reports—it reads relevant sections when answering.

**Try asking:**

- "What response rates did each project achieve?"
- "What are best practices for field data collection across these projects?"
- "What challenges were unique to Project B?"
- "Which project had the highest participant satisfaction?"
- "What would you recommend for a similar project based on these lessons?"

---

### Activity E — RAG with metadata filtering (advanced)

Run:

```bash
python examples/rag/05_rag_with_metadata.py
```

**What this script does:**

1. Extends the basic RAG system with **metadata filtering**:
   - Filter by specific project: "Only search Project A"
   - Filter by section type: "Only methodology sections"
   - Filter by document type: "Only findings and conclusions"
2. **Enhanced query interface**:
   - Allows users to specify filters before asking questions
   - Shows how many chunks match filters before retrieval
   - Displays filtered retrieval results
3. **Smart filtering**:
   - Natural language filter detection: "What did Project B say about response rates?"
   - Automatic project extraction from query
   - Section-specific searches

**Example with filtering:**

```text
You: [project:Conectividad] ¿Cuáles fueron las principales lecciones?

RAG System:
[Applying filter: project = Conectividad]
[Filtered chunks: 14 → 11 remaining]
[Retrieving from Conectividad only...]

Based on Conectividad documentation:

The main lessons were:
1. [Specific findings from Conectividad project...]
2. [Key insights...]
...
```

**Cómo usar los filtros - Ejemplos prácticos:**

| Tipo de pregunta | Sintaxis del filtro |
|------------------|---------------------|
| **Sin filtro** (busca en todos los proyectos) | `¿Cuáles fueron los principales desafíos?` |
| **Filtrar por proyecto específico** | `[project:Conectividad] ¿Qué lecciones se aprendieron?` |
| **Filtrar por otro proyecto** | `[project:Sisben] ¿Cuáles fueron los resultados?` |
| **Filtrar por sección** | `[section:Metodología] ¿Qué metodología se usó?` |
| **Combinar filtros** | `[project:Conectividad][section:Resultados] ¿Cuáles fueron los resultados?` |
| **Comparar proyectos** (sin filtro) | `¿Cuáles son las diferencias entre los proyectos?` |

**Ejemplos para tus proyectos:**

```text
# Pregunta específica sobre Conectividad
Your question: [project:Conectividad] ¿Qué desafíos enfrentó este proyecto?

# Pregunta específica sobre Sisben
Your question: [project:Sisben] ¿Cuáles fueron las lecciones aprendidas?

# Comparar ambos proyectos (sin filtro)
Your question: ¿Cuáles son las diferencias clave entre los proyectos?

# Usar coincidencia parcial
Your question: [project:Conect] ¿Qué se aprendió?
(funciona igual que [project:Conectividad])
```

**Nota:** Los filtros usan coincidencia parcial (`contains`), así que puedes escribir solo parte del nombre del proyecto.

**What to observe:**

- Does filtering improve answer relevance?
- How does chunk count change with filters?
- Can you ask cross-project questions without filters?
- Do filters work with natural language queries?

**What's happening:**

- Pre-filtering reduces search space before similarity search
- More targeted retrieval → more relevant context → better answers
- Combines structured filtering (metadata) with semantic search (embeddings)
- This is **hybrid retrieval**

**Key insight:** Metadata filtering is crucial for large document collections. It prevents the LLM from being overwhelmed with irrelevant context.

---

### Activity F — RAG evaluation and comparison (optional)

Run:

```bash
python examples/rag/06_rag_evaluation.py
```

**What this script does:**

1. Creates a **test set** of questions with known answers (ground truth)
2. **Compares three approaches**:
   - **Direct LLM** (no RAG, no document access)
   - **RAG with retrieval** (full pipeline)
   - **Oracle** (LLM with entire documents, not just chunks)
3. **Evaluates each approach**:
   - Factual accuracy (does answer match ground truth?)
   - Citation quality (are sources provided and correct?)
   - Completeness (is all relevant info included?)
   - Hallucination rate (does LLM make up information?)
4. Generates comparison report: `outputs/10_rag_evaluation_report.html`

**Test questions (examples):**

```text
1. What was the response rate in Project A?
   Ground truth: 68%, exceeding 65% target
   
2. What language barriers were encountered?
   Ground truth: Lack of indigenous language materials in Project B

3. Which project had scheduling conflicts with agricultural seasons?
   Ground truth: Project C

4. What did all three projects recommend for community engagement?
   Ground truth: Early partnership, culturally appropriate materials, flexible scheduling
```

**Evaluation metrics:**

| Metric | Definition | Good score |
|--------|------------|------------|
| **Accuracy** | % of factually correct answers | >90% |
| **Citation rate** | % of answers with valid citations | >80% |
| **Retrieval precision** | % of retrieved chunks that are relevant | >70% |
| **Hallucination rate** | % of answers with made-up facts | <5% |

**Sample results:**

```text
Approach         | Accuracy | Citations | Hallucinations
-----------------|----------|-----------|---------------
Direct LLM       | 25%      | 0%        | 60% (makes up details)
RAG (our system) | 92%      | 88%       | 3%
Oracle (all docs)| 95%      | 90%       | 2%
```

**What to observe:**

- How much does RAG improve over direct LLM?
- Where does RAG fail? (wrong retrieval? wrong generation?)
- Is RAG close to oracle performance?
- What types of questions are hardest for RAG?

**Key insight:** Evaluation reveals where your RAG system needs improvement—retrieval issues vs. generation issues require different fixes.

---

## Complete workflow example

**Scenario:** You've completed three community health intervention projects and want to build a knowledge base to inform future project design. You need to query: response rates, field challenges, best practices, budget considerations, and community engagement strategies.

**Step 1 — Prepare documents:**

- Collect final reports from all three projects (PDFs)
- Ensure documents are text-based (not scanned images)
- Place in `data/project_reports/`
  - `Project_A_Final_Report.pdf`
  - `Project_B_Evaluation.pdf`
  - `Project_C_Lessons_Learned.pdf`

**Step 2 — Convert to markdown:**

- Run `examples/01_pdf_to_markdown.py`
- Review converted markdown in `outputs/markdown/`
- Manually fix any extraction errors (tables, headers, lists)
- Validate that document structure is preserved

**Step 3 — Create embeddings:**

- Run `examples/02_create_document_chunks.py`
- Review chunking strategy: ~500 token chunks, section-based
- Check `outputs/07_document_chunks.csv`
- Verify metadata: all projects and sections labeled correctly

**Step 4 — Build vector database:**

- Run `examples/03_build_vector_database.py`
- Test retrieval with sample queries
- Validate that relevant chunks are retrieved
- Check similarity scores (should be >0.7 for good matches)

**Step 5 — Deploy RAG system:**

- Run `examples/04_rag_system_basic.py` for standard queries
- Run `examples/05_rag_with_metadata.py` for project-specific queries
- Save conversation history for record-keeping

**Step 6 — Query your knowledge base:**

**Response rates:**

```text
Q: What response rates did each project achieve?
A: Project A: 68% (target: 65%)
   Project B: 72% completion rate
   Project C: 63% after follow-up (initially 55%)
   [Citations from respective sections]
```

**Field challenges:**

```text
Q: What were the main field challenges across projects?
A: Common challenges included:
   - Trust-building and community skepticism (Project A)
   - Language barriers (Project B)
   - Scheduling conflicts with local seasons (Project C)
   All projects recommend early community partnership.
```

**Best practices:**

```text
Q: What are the top 3 lessons learned for future projects?
A: Based on all three projects:
   1. Start community engagement 6+ months before data collection
   2. Provide materials in local/indigenous languages
   3. Budget for transportation costs (40% of participants cited this)
```

**Project-specific:**

```text
Q: [Filter: Project_B] What made this project successful?
A: Project B achieved 72% completion through:
   - Partnerships with local health centers
   - Incentive structure aligned with community values
   - Flexible scheduling around harvest season
```

**Step 7 — Evaluate and iterate:**

- Run `examples/06_rag_evaluation.py`
- Review evaluation report
- Identify failure modes:
  - Poor retrieval? → Adjust chunking or k-value
  - Hallucinations? → Strengthen prompt instructions
  - Missing info? → Check source documents
- Re-run with improvements

**Step 8 — Document and share:**

- Export conversation history: `outputs/09_rag_conversations.jsonl`
- Generate summary report of key insights
- Share RAG system with project team
- Update knowledge base as new projects complete

---

## Common issues and solutions

**"PDF conversion has garbled text"**
→ PDF might be scanned (image-based). Docling runs OCR automatically; check the markdown in `outputs/markdown/` for recognition errors.

**"Chunks are too small/too large"**
→ Adjust chunking parameters in `02_create_document_chunks.py`. Try 300-800 token range.

**"Retrieval returns irrelevant chunks"**
→ Check similarity scores. If <0.6, increase k (retrieve more chunks) or rephrase query. Consider metadata filtering.

**"LLM hallucinates despite RAG"**
→ Strengthen prompt: "Answer ONLY from provided context. If not found, say 'Not available in documents.'" Consider using Claude for better instruction-following.

**"System is too slow"**
→ Reduce k (fewer chunks retrieved), cache embeddings, or use batch processing for multiple queries.

**"Answers lack detail"**
→ Increase chunk overlap (more context) or retrieve more chunks (higher k value, e.g., k=8).

**"Cross-project comparisons are weak"**
→ Ensure retrieval spans multiple projects (increase k or use metadata to ensure diversity). Add instruction: "Compare across all available projects."

**"Database file is too large"**
→ Use a proper vector database (Pinecone, Weaviate, Chroma) instead of CSV/pickle. These are optimized for embeddings.

---

## Mental model: When to use RAG

```text
Need to query project-specific information?
├─ YES → Use RAG
│   ├─ Single project focus? → Use metadata filtering
│   └─ Cross-project analysis? → Use full retrieval
└─ NO → Use direct LLM prompting

Document collection size:
├─ 1-3 documents → RAG is overkill, use direct prompting with full text
├─ 3-10 documents → RAG is helpful
└─ 10+ documents → RAG is essential

Query types:
├─ Factual lookup (response rates, dates, numbers) → RAG excels
├─ Synthesis across documents → RAG + careful prompting
├─ Creative tasks (brainstorming, ideation) → Direct LLM
└─ Opinion/judgment → Hybrid (RAG for facts, LLM for reasoning)
```

---

## What you've learned

- ✅ Convert PDFs to markdown for LLM processing
- ✅ Create document chunks with metadata
- ✅ Generate embeddings and build vector databases
- ✅ Implement retrieval systems (semantic search)
- ✅ Build end-to-end RAG pipelines
- ✅ Query project knowledge with citations
- ✅ Use metadata filtering for targeted retrieval
- ✅ Evaluate RAG performance vs. direct prompting
- ✅ Understand trade-offs: RAG vs. fine-tuning vs. direct prompting

---

## Discussion prompts

Use these to reflect on your RAG implementation:

- Which queries worked best with RAG? Which didn't?
- How did retrieval quality affect answer quality?
- Are there chunks that should be split differently?
- Should we use multi-query retrieval (generate multiple query variants)?
- How do we handle contradictions across projects?
- What metadata would improve filtering?
- How do we keep the knowledge base updated as projects evolve?
- Should we implement user feedback to improve retrieval?
- How do we validate that citations are accurate?

---

## Key takeaways

1. **RAG gives LLMs access to your documents**: Without retraining, LLMs can reference project-specific knowledge
2. **Retrieval quality is critical**: Bad retrieval = bad answers, even with good LLMs
3. **Chunking strategy matters**: Section-based > fixed-size for structured documents
4. **Metadata enables targeted search**: Filter by project, section, or document type
5. **Always ground answers in context**: Prevent hallucinations by requiring citations
6. **Evaluate systematically**: Compare RAG to baselines (direct LLM, oracle)
7. **Iterate based on failure modes**: Fix retrieval issues separately from generation issues

---

## Next steps and extensions

**If you want to go further:**

### Improve retrieval

- **Hybrid search**: Combine semantic (embeddings) + keyword (BM25) search
- **Query expansion**: Generate multiple query variants for better coverage
- **Re-ranking**: Use a second model to re-score retrieved chunks
- **Metadata enrichment**: Add tags, themes, project dates to chunks

### Scale the system

- **Vector databases**: Use Pinecone, Weaviate, or Chroma for production
- **Batch processing**: Pre-compute answers to common questions
- **Caching**: Store previous queries and answers
- **API wrapper**: Deploy as REST API for team access

### Advanced features

- **Multi-turn conversations**: Maintain chat history and context
- **Source diversity**: Ensure chunks from multiple projects in retrieval
- **Answer validation**: Have a second LLM verify citations
- **User feedback loop**: Let users mark good/bad answers to improve retrieval

### Evaluation

- **Create larger test sets**: 50-100 questions with ground truth
- **Human evaluation**: Have domain experts rate answer quality
- **A/B testing**: Compare different chunking/retrieval strategies
- **Error analysis**: Categorize failure modes (retrieval vs. generation)

### Integration

- **Research dashboards**: Embed RAG in internal research tools
- **Slack/Teams bot**: Allow team members to query knowledge base via chat
- **Report generation**: Automatically generate cross-project synthesis reports
- **Live updates**: Re-index as new project documents are added

---

## Resources

- **RAG Fundamentals**: <https://www.pinecone.io/learn/retrieval-augmented-generation/>
- **LangChain RAG Tutorial**: <https://python.langchain.com/docs/use_cases/question_answering/>
- **Vector Databases Comparison**: <https://www.pinecone.io/learn/vector-database/>
- **OpenAI Embeddings Guide**: <https://platform.openai.com/docs/guides/embeddings>
- **Chunking Strategies**: <https://www.pinecone.io/learn/chunking-strategies/>
- **RAG Evaluation**: <https://docs.llamaindex.ai/en/stable/examples/evaluation/>
- **PDF Processing**: <https://docling-project.github.io/docling/>

---

## Tools to explore

**Vector databases:**

- **Pinecone**: Managed, scalable, easy setup
- **Weaviate**: Open-source, feature-rich
- **Chroma**: Lightweight, good for prototyping
- **FAISS**: Facebook's library, fast but lower-level

**RAG frameworks:**

- **LangChain**: Comprehensive, lots of integrations
- **LlamaIndex**: Focused on document indexing
- **Haystack**: Production-ready, NLP focus

**Evaluation tools:**

- **RAGAS**: RAG evaluation metrics
- **TruLens**: RAG observability and evaluation
- **LangSmith**: LangChain's evaluation platform

---

**Congratulations!** You've built a complete RAG system for project knowledge management. You can now turn any collection of documents into a queryable knowledge base that LLMs can use to provide accurate, cited answers to your research questions.
