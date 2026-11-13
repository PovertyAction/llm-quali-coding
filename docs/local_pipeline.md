# Local pipeline: code a focus group without sending the transcript anywhere

This guide runs qualitative coding on a laptop. Open-weights models find and
code passages, Presidio replaces names and numbers, and only the de-identified
passages go to Claude, through Claude Code. No OpenAI or Anthropic API key is
needed.

All data in `data/fictitious/` is invented. Use the same steps on real
transcripts only after the IRB protocol and the IPA AI Usage Guidelines allow
it: a de-identified transcript is not automatically Internal data. Ask
<support@poverty-action.org> when unsure.

## What runs where

  | Step                      | Tool                                         | Where  |
  | ------------------------- | -------------------------------------------- | ------ |
  | 1. Split and de-identify  | Presidio, spaCy `es_core_news_md`, name list | Laptop |
  | 2. Filter and code        | `granite-embedding-97m-multilingual-r2`      | Laptop |
  | 3. Code the same passages | Claude Code, reading the de-identified CSV   | Claude |
  | 4. Compare with a person  | scikit-learn: agreement, Cohen's kappa       | Laptop |

## Setup, once

You need [uv](https://docs.astral.sh/uv/) and
[just](https://just.systems/). The main [README](../README.md) shows how to
install them. Then, in a terminal opened in the repository folder:

```bash
just venv
uv run python -c "from src.local_embeddings import embed_texts; embed_texts(['hola'])"
```

The first line installs Python and the packages. The second downloads the
embedding model (about 210 MB) into the Hugging Face cache. After that, steps
1, 2, and 4 work offline.

## Run it

```bash
just local-deid       # step 1: outputs/local/01_passages_deid.csv and 01_findings.csv
just local-classify   # step 2: outputs/local/02_coded.csv
                      # step 3: ask Claude Code, see below
just local-compare    # step 4: agreement with data/fictitious/codigos_humanos.csv
```

Each command takes under a minute on a laptop CPU once the models are
downloaded. Everything they write goes to `outputs/local/`, which Git ignores.

### Step 1: read the findings

`01_passages_deid.csv` has one row per participant turn: the speaker label,
the moderator's question, and what the participant said. Names, phone numbers,
emails, and places are replaced with tags such as `<PERSONA>`, in the question
as well as the answer.

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

An embedding model turns each passage, and each theme definition in the
codebook, into a list of numbers. Texts with similar meaning get similar
numbers, so the script can score how close a passage is to each theme.

Each passage gets a relevance score: its similarity to the closest theme.
Passages under `THRESHOLD` (0.78) are coded `Fuera de tema`; the rest get the
most similar theme. `02_coded.csv` keeps both: `nearest_theme` is the closest
theme whatever the score, and `embedding_theme` is the code after the filter.

On the fictitious data this agrees with the human codes on 15 of the 20
sampled passages (75 percent, Cohen's kappa 0.70).

The threshold is a trade-off:

- **Lower** keeps more passages. Small talk slips through and gets a theme it
  does not deserve, but you see it when you read the results.
- **Higher** keeps only strong matches. Real passages that mention a theme
  indirectly are dropped, and nobody reads the discard pile unless told to.

When unsure, go lower. Step 4 prints the agreement for each candidate value.

### Step 3: ask Claude Code

Start `claude` in the repository root and paste:

```text
Read outputs/local/01_passages_deid.csv and data/fictitious/codebook_ficticio.json.
For every passage, choose exactly one theme short name from the codebook, or
"Fuera de tema" if none fits. Write outputs/local/03_claude_codes.csv with the
columns passage_id and theme. Use only the question and the passage text; do
not guess who anyone is. Do not open any other file in data/ or outputs/.
```

Claude reads the de-identified passages, not `02_coded.csv`, so it does not
see the codes from step 2 and the two stay independent.

Read the plan and the file Claude writes before you accept.
`.claude/settings.json` also tells Claude Code not to read the raw transcript,
the name list, or the findings files. Treat that as a guardrail, not a
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

For the embeddings it also prints the agreement at each candidate `THRESHOLD`.
With 20 passages, one passage moves agreement by 5 points, so a difference of
one row in that table is not evidence. Code more passages by hand before you
trust a threshold.

## Using your own data

1. Keep the raw transcript outside any folder Claude Code opens, or add it to
   the deny list in `.claude/settings.json`.
2. Put the team roster in a name list; keep it local.
3. Code 20 to 30 passages by hand before you look at any model output.
4. Point the paths at the top of the scripts in `examples/local/` to your
   files. The scripts expect speaker labels such as `MODERADOR:` and
   `PARTICIPANTE 1:` at the start of a line.
5. Retune `THRESHOLD` in `examples/local/02_filter_classify.py` with the table
   from step 4.

## Trying another embedding model

Copy `.env.example` to `.env` and set `LOCAL_EMBEDDING_MODEL` to any
[Sentence Transformers](https://sbert.net/) model on Hugging Face, for example
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (about 1.1 GB).

Every model scores on its own scale. The default model puts all passages
between about 0.70 and 0.90; the mpnet model spreads them from 0.10 to 0.70.
A threshold tuned for one model is wrong for another, so rerun steps 2 and 4
and pick a new `THRESHOLD` from the table.

## If something goes wrong

- **`Missing outputs/local/...`**: run the earlier step first. The steps go in
  order: `local-deid`, `local-classify`, `local-compare`.
- **`ModuleNotFoundError: No module named 'src'`**: run `just venv` again, and
  run commands from the repository folder.
- **The first run of a step is slow or fails without internet**: it is
  downloading a model. Later runs work offline.
- **Accents look wrong when you open a CSV in Excel**: the files are UTF-8.
  Open them in VS Code, or in Excel with Data > From Text/CSV and choose UTF-8.
