# LLM Qualitative Coding

Training materials for using Large Language Models (LLMs) to code and analyze
qualitative data, such as interview and focus group transcripts.

You do not need LLM experience. Basic Python helps, but every step is a
command you can copy and paste.

## Pick your starting point

| I want to...                                             | Start here                                                         | API key needed?     |
| -------------------------------------------------------- | ------------------------------------------------------------------ | ------------------- |
| Learn the basics: call an LLM from Python                | [Session 01](docs/session_01_setup.md) ([español](docs/session_01_setup_es.md)) | Anthropic or OpenAI |
| Understand embeddings and search documents by meaning    | [Session 02](docs/session_02_embeddings_rag.md) ([español](docs/session_02_embeddings_rag_es.md)) | OpenAI |
| Code transcripts with embeddings and LLMs                | [Session 03](docs/session_03_quali_coding.md)                      | OpenAI              |
| Ask questions of a set of PDF reports (RAG)              | [Session 04](docs/session_04_rag_demo.md) ([español](docs/session_04_rag_demo_es.md)) | OpenAI |
| Code transcripts on my laptop, sending nothing to an API | [Local pipeline](docs/local_pipeline.md)                           | None                |
| Code my own transcripts in a browser, without writing code | [Web app](#use-the-web-app)                                      | Optional            |

New to all of this? Do the setup below, then the sessions in order. Each one
builds on the one before. The local pipeline stands on its own.

## Setup

Do this once. It takes about ten minutes.

### 1. Install the tools

You need three programs: [Git](https://git-scm.com/) to download the
materials, [uv](https://docs.astral.sh/uv/) to install Python and its
packages, and [just](https://just.systems/) to run short commands such as
`just venv`.

Open a terminal (PowerShell on Windows, Terminal on macOS) and run:

```bash
# Windows
winget install Git.Git astral-sh.uv Casey.Just

# macOS, with Homebrew (https://brew.sh)
brew install git uv just
```

Close the terminal and open a new one so it finds the new programs.

You also need a code editor. We suggest
[VS Code](https://code.visualstudio.com/) or [Positron](https://positron.posit.co/).

### 2. Download the materials

```bash
git clone https://github.com/PovertyAction/llm-quali-coding.git
cd llm-quali-coding
```

Open the `llm-quali-coding` folder in your editor. Run every later command
from a terminal inside this folder (in VS Code: Terminal > New Terminal).

### 3. Install Python and the packages

```bash
just venv
```

This installs the right Python version and every package into a `.venv`
folder inside the project. Nothing else on your computer changes. The first
run downloads a few gigabytes and can take several minutes.

### 4. Add your API key

Skip this step if you only want the [local pipeline](docs/local_pipeline.md).

```bash
cp .env.example .env
```

Open the new `.env` file in your editor and replace `sk-...` or `sk-ant-...`
with your own key. Delete the line for a key you do not have. Sessions 02 to
04 need the OpenAI key, because they use OpenAI embeddings.

- Anthropic keys: <https://console.anthropic.com/settings/keys>
- OpenAI keys: <https://platform.openai.com/api-keys>

An API key works like a password that is billed to your account. Never paste
it into a script, a chat, or an email. Git ignores the `.env` file, so the key
stays on your computer.

### 5. Check that it works

```bash
uv run python examples/basics/test_connection.py
```

You should see `Connection successful!` and a short greeting from the model.
If you do, setup is done: open [Session 01](docs/session_01_setup.md).

Starting a command with `uv run` makes it use the project's Python and
packages. The session guides show another way, activating the environment
first. Both work.

## Try the local pipeline (no API key)

After setup steps 1 to 3:

```bash
just local-deid       # split a fictitious focus group and remove names
just local-classify   # code each passage with a model on your laptop
just local-compare    # compare the codes with a person's codes
```

The [local pipeline guide](docs/local_pipeline.md) explains each step and how
to add Claude Code as a second coder.

## Use the web app

The web app runs the coding steps from Session 03 and the local pipeline in
your browser, on your own transcripts and your own codebook. After setup steps
1 to 3:

```bash
just app
```

Without just: `uv run streamlit run app/main.py`. Your browser opens at
<http://localhost:8501>. Only your computer can open that address. Press
Ctrl+C in the terminal to stop the app.

Work through the pages in the left menu in order. A check mark appears beside
a step once it is done, and each page ends with a link to the next one:

1. **Project and models.** Create a project, or load the example project (the
   invented focus group). Choose where the language model runs:
   - *On this computer*: install [Ollama](https://ollama.com) or
     [LM Studio](https://lmstudio.ai) and download a model. Nothing leaves your
     computer. Small models are slow without a graphics card.
   - *Cloud API*: Anthropic or OpenAI, with your own API key.
   - *Institution cloud*: Azure OpenAI or AWS Bedrock, if your organization
     gives you access.
2. **Study inputs.** Write the research question and study context. Build the
   codebook in the table: a name and a definition for each theme, and
   optionally when to include it, when to exclude it, and example quotes. You
   can load a codebook from JSON or CSV. Choose what counts as a passage and
   whether each passage gets one theme or a yes/no for every theme.
3. **Transcripts.** Add as many transcripts as you like (`.md`, `.txt`,
   `.docx`, `.pdf`). The app works best when each turn is a line that starts
   with `SPEAKER:`. De-identify them, then read each passage next to the
   original. Only new or changed transcripts are prepared again, so adding a
   transcript does not disturb the ones already coded.
4. **Run.** Tick the steps and run them on every transcript: a quick first
   pass by similarity (no language model), code with a language model, code
   again with a second model, flag non-verbal cues, propose themes, translate.
   The page says where the text goes before you run.
5. **Results.** Browse the coded passages, count passages per theme, add
   proposed themes to the codebook, and download a CSV or the whole project.
6. **Check against human codes.** Upload a person's codes for a sample of
   passages to see the agreement and Cohen's kappa. Do this before trusting
   the model's codes.

Languages:

- The interface is available in English, Spanish and French. Choose the
  language at the bottom of the left menu; the app remembers it. The first
  time, it follows your browser's language.
- The example project comes in the same three languages. Pick the language of
  the example on the first page.
- Your transcripts can be in any language the models you choose can read.
  Tell the app the language of the placeholders (`<PERSON>`, `<PERSONA>`,
  `<PERSONNE>`) on the Transcripts page, and the language for the model's
  reasons on the Study inputs page.
- To add an interface language, copy `app/locales/es.json` to
  `app/locales/<code>.json`, set `"_name"`, and translate the values. Run
  `just test`: it lists any string the new file is missing.

Good to know:

- Each project is a folder in `outputs/projects/`. Git ignores it. Close the
  app and reopen it later: your inputs and results are still there.
- The app saves every answer a model gives. If a run stops or one transcript
  fails, run it again: only the missing answers are requested, so you do not
  pay twice.
- API keys are kept for the session only. They are never saved in the project.
  The app reads the keys in your `.env` file if you have one.
- The left menu always shows whether text stays on your computer or which
  service receives it. The model list shows the data classification each
  service is approved for. A run that sends passages to a cloud service says
  exactly what is sent and asks you to confirm the classification first.
- After preparing a transcript, read it and mark it as reviewed. The Run page
  warns when a transcript headed for a cloud service has not been marked.
- If you change the codebook or the model after a run, the app tells you the
  codes on screen are out of date until you run again.
- The project download leaves out the original transcripts, the roster and the
  list of replaced names.
- Coding with embeddings needs a relevance threshold that fits the embedding
  model. Page 6 shows the agreement for each value.
- A model on your own computer is slow without a graphics card: expect up to a
  minute per passage on a laptop. Keep the computer awake during a run.
- "Propose themes" sends a whole transcript in one request. Local model
  servers often read only the first few pages by default (Ollama: 4,096
  tokens) and drop the rest without a warning. Raise the context length in
  the server's settings, or use a cloud model for that step.

## What is in this repository

```text
llm-quali-coding/
├── docs/                         # Guides: read these in order
│   ├── session_01_setup.md       # Setup and first LLM API calls (English)
│   ├── session_02_embeddings_rag.md  # Embeddings, similarity search, a first RAG
│   ├── session_03_quali_coding.md  # Qualitative coding with embeddings and LLMs
│   ├── session_04_rag_demo.md    # RAG over PDF reports, with evaluation
│   ├── session_0X_..._es.md      # Sessions 01, 02 and 04 in Spanish
│   └── local_pipeline.md         # Coding on the laptop with no API key
├── examples/                     # Scripts you run
│   ├── basics/                   # Session 01: connection test, chat, streaming, tools
│   ├── embeddings/               # Session 02: embeddings, similarity search, mini-RAG
│   ├── 01_translate_transcript.py ... 07_model_comparison.py   # Session 03
│   ├── rag/                      # Session 04: PDF to text, vector database, RAG, evaluation
│   └── local/                    # Local pipeline: 01_deidentify, 02_filter_classify, 04_compare
├── app/                          # The web app (`just app`)
├── src/                          # Code the scripts and the app share; you rarely edit this
├── tests/                        # Checks for the pipeline and the app (`just test`)
├── data/                         # Small sample files; nothing sensitive
│   ├── ipa-best-bets-2025.md     # Document translated in Session 01
│   ├── sample_docs/, exercise/   # Short documents for Session 02
│   ├── sample_transcripts/       # Transcripts for Session 03
│   ├── themes/                   # Codebooks for Session 03
│   ├── project_reports/          # PDF reports for Session 04
│   └── fictitious/               # Invented focus group for the local pipeline
├── outputs/                      # Created when you run scripts; Git ignores it
├── .env.example                  # Template for your .env file
├── Justfile                      # The `just` commands
└── pyproject.toml                # The list of Python packages
```

## Before you use real data

The sample data here is public or invented. Real transcripts usually contain
names and other personal information.

- Sessions 01 to 04 send text to OpenAI or Anthropic. Do not send
  confidential data or anything that identifies a research participant.
- The local pipeline keeps the transcript on your laptop, and it still needs
  your IRB protocol to allow the analysis.
- The web app keeps everything on your laptop only when both models on page 1
  are set to "On this computer". With a cloud model, the de-identified
  passages are sent to that service. De-identification misses things: read
  the passages before you run a cloud step.
- Read the IPA AI Usage Guidelines (Box: `IPA_IT_Resources/Policies/AI - Claude`)
  and ask <support@poverty-action.org> when unsure.

## If something goes wrong

| What you see                                    | What to do                                                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `just` or `uv` is "not recognized" or "not found" | Close the terminal, open a new one, and try again. If it still fails, repeat setup step 1.   |
| `No API keys found`                             | Check that `.env` exists in the project folder and holds your key (setup step 4).            |
| `ModuleNotFoundError`                           | Run `just venv` again, and run commands from the project folder with `uv run` in front.      |
| `Missing outputs/...`                           | Run the earlier step first. Each guide lists the steps in order.                             |
| A step is slow the first time                   | It is downloading a model. Later runs are fast and work offline.                             |

Each guide has its own troubleshooting section for the scripts it covers.

## What you will learn

- How an LLM API call works, and how to keep your key safe
- What embeddings are and how they find passages about a topic
- How retrieval augmented generation (RAG) answers questions from your own
  documents, and how to check its answers
- Several ways to code transcripts: by similarity to a codebook, by asking an
  LLM, and by letting an LLM propose themes
- How to check model codes against a person's codes before trusting them
