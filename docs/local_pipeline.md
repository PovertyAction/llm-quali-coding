# Local pipeline: code a focus group without sending the transcript anywhere

This guide runs qualitative coding on a laptop. Open-weights models find and
code passages, Presidio replaces names and numbers, and only the de-identified
passages go to Claude, through Claude Code. No OpenAI or Anthropic API key is
needed.

All data in `data/fictitious/` is invented. Use the same steps on real
transcripts only after the IRB protocol and the IPA AI Usage Guidelines allow
it: a de-identified transcript is not automatically Internal data. Ask
support@poverty-action.org when unsure.

## What runs where

  | Step                       | Tool                                         | Where          |
  | -------------------------- | -------------------------------------------- | -------------- |
  | 1. Split and de-identify   | Presidio, spaCy `es_core_news_md`, name list | Laptop         |
  | 2. Filter and code         | `paraphrase-multilingual-mpnet-base-v2`      | Laptop         |
  | 3. Code the same passages  | Claude Code, reading the de-identified CSV   | Claude         |
  | 4. Compare with a person   | scikit-learn: agreement, Cohen's kappa       | Laptop         |

## Setup, once

```bash
just venv
uv run python -c "from src.local_embeddings import embed_texts; embed_texts(['hola'])"
```

The second line downloads the embedding model (about 1.1 GB) into the Hugging
Face cache. After that, steps 1, 2, and 4 work offline.

## Run it

```bash
just local-deid       # outputs/local/01_passages_deid.csv and 01_findings.csv
                      # optional: just local-deid-gliner, see below
just local-classify   # outputs/local/02_coded.csv
just local-compare    # agreement with data/fictitious/codigos_humanos.csv
```

Each step takes under a minute on a laptop CPU once the models are downloaded.

### Step 1: read the findings

`01_findings.csv` lists every replacement: entity type, the text found, a
score, and which recognizer found it. It holds the original names, so it stays
on the laptop. Check two things:

- **Misses.** spaCy's Spanish model does not find every name. The name list in
  `lista_participantes.txt` (the roster a team already has) catches the rest.
  Remove a name from the list and rerun to see what slips through.
- **Over-redaction.** The model tags some ordinary words as names. That loses
  meaning but does not leak anything.

### Optional: a second detector, GLiNER-PII

[GLiNER-PII](https://huggingface.co/nvidia/gliner-PII) is an open-weights
model from NVIDIA that reads the whole sentence instead of matching patterns.
It downloads once (about 1.7 GB), then runs on the laptop.

```bash
just local-deid-gliner   # outputs/local/01_passages_deid_gliner.csv and 01_findings_gliner.csv
```

On the fictitious focus group it found every planted name, phone number, and
email without the name list. It also replaced "Mi esposo", "Yo", and
"tesorera", which the coding step needs. Its model card lists English only.
Compare the two findings files; step 2 reads the Presidio file. The threshold
(0.3, from the model's evaluation) is an argument to `deidentify_gliner()`.
Raising it cuts over-redaction and raises the risk of misses.

### Step 2: filter and code with embeddings

Each passage gets a relevance score: its similarity to the closest codebook
theme. Passages under the threshold (0.25, set low on purpose) are coded
`Fuera de tema`; the rest get the most similar theme. On the fictitious data
this agrees with the human codes on about 80 percent of the sample. Most
misses are small talk the filter let through.

### Step 3: ask Claude Code

Start `claude` in the repository root and paste:

```text
Read outputs/local/02_coded.csv and data/fictitious/codebook_ficticio.json.
For every passage, choose exactly one theme short name from the codebook, or
"Fuera de tema" if none fits. Write outputs/local/claude_codes.csv with the
columns passage_id and theme. Use only the passage text; do not guess who
anyone is. Do not open any other file in data/ or outputs/.
```

Read the plan and the file Claude writes before you accept.
`.claude/settings.json` also tells Claude Code not to read the raw transcript,
the name list, or the findings file. Treat that as a guardrail, not a
guarantee: the safest place for a raw transcript is a folder Claude Code never
opens.

### Step 4: compare

`just local-compare` prints, for the embeddings and for Claude:

- **Agreement:** the share of the 20 sampled passages where the model matches
  the person.
- **Cohen's kappa:** agreement corrected for chance. Above 0.8 is strong; under
  0.6 means the codebook or the model needs work.
- **A confusion table and the list of disagreements.** Read each one. Decide
  whether the definition was unclear (fix the codebook), the model was wrong
  (keep the human code), or the human code was wrong.

## Using your own data

1. Keep the raw transcript outside any folder Claude Code opens, or add it to
   the deny list in `.claude/settings.json`.
2. Put the team roster in a name list; keep it local.
3. Code 20 to 30 passages by hand before you look at any model output.
4. Retune `THRESHOLD` in `examples/local/02_filter_classify.py`.
