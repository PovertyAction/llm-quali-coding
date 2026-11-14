"""The qualitative coding pipeline as plain functions, used by the web app.

Each step takes and returns a table of passages (one row per passage), so the
steps can be run one transcript at a time or in a batch. The steps that call a
language model take a `complete(system, user) -> str` function instead of a
client, which keeps them independent of the provider and easy to test.

A project is a folder:

    project.json            settings, research question, codebook, prompts
    roster.txt              participant names for de-identification (stays local)
    llm_cache.jsonl         every model answer, so a rerun does not pay twice
    transcripts/            the original files (stay local)
    results/<file name>/
        passages.csv        de-identified passages
        findings.csv        what was replaced; holds the original names
        coded.csv           passages with the codes from every step that ran
        themes.json         themes the model proposed for this transcript
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import warnings
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .chunking import chunk_text_simple, make_chunks, parse_speakers
from .coding import (
    Theme,
    add_theme_similarity_columns,
    classify_by_max_theme,
    theme_prompt_text,
    themes_from_data,
)
from .llm_client import complete_with_usage
from .similarity import dot_similarity

Complete = Callable[[str, str], str]  # (system prompt, user message) -> answer
Embed = Callable[..., list[list[float]]]  # (texts, query=False) -> vectors
Clean = Callable[[str], tuple[str, list[dict]]]  # text -> (clean text, findings)
Progress = Callable[[float, str], None]  # (fraction done, what is running)

TRANSCRIPT_TYPES = [".md", ".txt", ".docx", ".pdf"]
MODERATOR_LABELS = [
    "MODERADOR",
    "MODERATOR",
    "INTERVIEWER",
    "ENTREVISTADOR",
    "ANIMATEUR",
    "ANIMATRICE",
    "MODÉRATEUR",
    "MODÉRATRICE",
]
UNPARSED = "(could not read answer)"
# Files that hold original names or text. They are never sent to a model and
# are left out of the project download.
LOCAL_ONLY = {
    "findings.csv",
    "roster.txt",
    "llm_cache.jsonl",
    "prepared.txt",
    "reviewed.txt",
    "last_run.json",
}
# Settings that decide the codes, by the kind of step that reads them. A change
# to one of them means the codes from that kind of step are out of date.
CODING_KEYS = {
    "all": ["themes", "off_topic"],
    "embeddings": ["threshold", "research_question", "embed_backend", "embed_model"],
    "model": [
        "coding_mode",
        "prompts",
        "research_question",
        "study_context",
        "output_language",
        "backend",
        "model",
    ],
}
# Settings that change the passages, so a change means preparing again.
PREP_KEYS = [
    "unit",
    "moderator_labels",
    "participant_label",
    "detector",
    "detector_threshold",
    "proper_nouns_only",
    "keep_terms",
    "placeholders",
]

DEFAULTS: dict = {
    # models
    "backend": "local",
    "model": "",
    "model2": "",  # second model on the same backend, for comparison
    "models": {},  # the model last chosen for each backend
    "steps": ["embeddings", "llm"],  # steps ticked on the Run page
    "base_url": "",
    "azure_endpoint": "",
    "api_version": "",
    "aws_region": "",
    "aws_profile": "",
    "embed_backend": "local",
    "embed_model": "",
    "workers": 4,
    # study inputs
    "research_question": "",
    "study_context": "",
    "output_language": "English",
    "unit": "turn",  # turn | question | fixed
    "moderator_labels": MODERATOR_LABELS,
    "participant_label": "PARTICIPANT",
    "off_topic": "Off topic",
    # Every embedding model scores on its own scale, so this needs tuning:
    # the human-codes check prints the agreement for each candidate value.
    "threshold": 0.78,
    "n_themes": "8-15",
    "coding_mode": "single",  # single | multi
    "themes": [],
    "prompts": {},
    # de-identification
    "detector": "gliner",  # gliner | presidio | none
    "detector_threshold": 0.5,
    "placeholders": "es",  # language of the tags: es, en or fr
    "proper_nouns_only": True,  # GLiNER2-PII: leave lowercase words alone
    "keep_terms": [],  # names of large places and other words to leave in
}

# The instructions sent to the model for each step. A project can override any
# of them. {placeholders} are filled in by _fill(); other braces are left alone.
PROMPTS = {
    "code_single": (
        "You are a qualitative researcher coding interview and focus group "
        "passages against a codebook.\n"
        "{study}\n"
        "CODEBOOK\n{codebook}\n\n"
        "Assign the passage to the one theme it most substantively discusses. "
        "The moderator question is context only. If the passage fits no theme, "
        'use "{off_topic}".\n'
        'Reply with JSON only: {"theme": "<exact theme name, or {off_topic}>", '
        '"rationale": "<one sentence in {output_language}>"}'
    ),
    "code_multi": (
        "You are a qualitative researcher coding interview and focus group "
        "passages against a codebook.\n"
        "{study}\n"
        "THEME\n{theme}\n\n"
        "Decide whether the passage substantively discusses the theme. "
        "The moderator question is context only.\n"
        "Reply with one word: YES or NO."
    ),
    "themes": (
        "You are a qualitative researcher doing inductive coding of an "
        "interview or focus group transcript.\n"
        "{study}\n"
        "Read the whole transcript and identify the {n_themes} main themes "
        "that capture the breadth of the discussion. Write theme names and "
        "definitions in {output_language}; quotes stay in the transcript's "
        "language.\n"
        'Reply with JSON only: a list of {"name": "<short name>", '
        '"definition": "<one or two sentences>", '
        '"examples": "<one or two short quotes>"}'
    ),
    "nonverbal": (
        "You are a qualitative researcher extracting non-verbal cues from "
        "transcript notes.\n"
        "Detect whether the passage records any explicit non-verbal cue "
        "(laughter, pauses, crying, confusion, gestures).\n"
        'Reply with JSON only: {"any_cues": "YES" or "NO", '
        '"cue_type": "<short label in {output_language}, or empty>"}'
    ),
    "translate": (
        "You are a translator of interview and focus group transcripts.\n"
        "Translate the text into {output_language}. Keep placeholders such as "
        "<PERSONA> or <PERSON> and the formatting unchanged.\n"
        "Reply with the translation only."
    ),
}


# --- project folder ---------------------------------------------------------


def load_settings(project_dir: Path) -> dict:
    """Read project.json, filling in defaults for anything it does not set."""
    path = project_dir / "project.json"
    saved = json.loads(path.read_text("utf-8")) if path.exists() else {}
    return {**DEFAULTS, **saved}


def save_settings(project_dir: Path, settings: dict) -> None:
    """Write project.json. API keys are not part of the settings."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_transcripts(project_dir: Path) -> list[Path]:
    """List the project's transcript files."""
    folder = project_dir / "transcripts"
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in TRANSCRIPT_TYPES)


def result_dir(project_dir: Path, transcript: Path) -> Path:
    """Folder holding the results for one transcript."""
    return project_dir / "results" / transcript.name


def prep_fingerprint(project_dir: Path, settings: dict, transcript: Path) -> str:
    """Hash of everything that decides a transcript's passages.

    Covers the file, the roster and the passage and de-identification settings.
    """
    roster = project_dir / "roster.txt"
    digest = hashlib.sha256(transcript.read_bytes())
    digest.update(roster.read_bytes() if roster.exists() else b"")
    digest.update(json.dumps([settings[k] for k in PREP_KEYS]).encode())
    return digest.hexdigest()


def needs_prepare(project_dir: Path, settings: dict, transcript: Path) -> bool:
    """Whether a transcript is new, or changed since it was last prepared."""
    out = result_dir(project_dir, transcript)
    stamp = out / "prepared.txt"
    if not (out / "passages.csv").exists() or not stamp.exists():
        return True
    return stamp.read_text("utf-8") != prep_fingerprint(
        project_dir, settings, transcript
    )


def is_reviewed(project_dir: Path, transcript: Path) -> bool:
    """Whether a person marked these exact passages as read after de-identifying."""
    out = result_dir(project_dir, transcript)
    mark, stamp = out / "reviewed.txt", out / "prepared.txt"
    return (
        mark.exists()
        and stamp.exists()
        and mark.read_text("utf-8") == stamp.read_text("utf-8")
    )


def set_reviewed(project_dir: Path, transcript: Path, reviewed: bool) -> None:
    """Record, or withdraw, that a person read the prepared passages."""
    out = result_dir(project_dir, transcript)
    if reviewed:
        stamp = (out / "prepared.txt").read_text("utf-8")
        (out / "reviewed.txt").write_text(stamp, "utf-8")
    else:
        (out / "reviewed.txt").unlink(missing_ok=True)


def coding_fingerprint(settings: dict, steps: list[str] | None = None) -> str:
    """Hash of the inputs the given steps read, to tell when codes are out of date.

    A run of the similarity pass alone does not depend on the language model,
    so choosing another model afterwards does not make its codes stale.
    """
    steps = list(STEPS) if steps is None else steps
    keys = list(CODING_KEYS["all"])
    if "embeddings" in steps:
        keys += CODING_KEYS["embeddings"]
    if any(step != "embeddings" for step in steps):
        keys += CODING_KEYS["model"]
    raw = json.dumps([settings.get(k) for k in sorted(set(keys))], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def project_status(project_dir: Path, settings: dict) -> dict:
    """Where a project stands: counts the app uses to show what is done.

    Keys: themes, transcripts, prepared (up to date), stale (prepared before a
    change), reviewed, coded, passages.
    """
    status = {"themes": len(settings["themes"]), "transcripts": 0, "prepared": 0}
    status |= {"stale": 0, "coded": 0, "passages": 0, "reviewed": 0}
    for transcript in list_transcripts(project_dir):
        out = result_dir(project_dir, transcript)
        status["transcripts"] += 1
        if not (out / "passages.csv").exists():
            continue
        if needs_prepare(project_dir, settings, transcript):
            status["stale"] += 1
        else:
            status["prepared"] += 1
        status["passages"] += len(read_table(out / "passages.csv"))
        status["coded"] += (out / "coded.csv").exists()
        status["reviewed"] += is_reviewed(project_dir, transcript)
    return status


def estimate_requests(status: dict, settings: dict, steps: list[str]) -> int:
    """Upper bound on the model requests a run makes, before reuse of old answers."""
    passages, themes = status["passages"], max(status["themes"], 1)
    per_code = passages * (themes if settings["coding_mode"] == "multi" else 1)
    cost = {"translate": passages, "llm": per_code, "llm2": per_code}
    cost |= {"nonverbal": passages, "themes": status["prepared"] + status["stale"]}
    return sum(cost.get(step, 0) for step in steps)


def read_table(path: Path) -> pd.DataFrame:
    """Read a passages or coded CSV, keeping empty cells as empty text."""
    return pd.read_csv(path, keep_default_na=False)


def load_coded(project_dir: Path) -> pd.DataFrame:
    """All coded passages in the project, with a `transcript` column."""
    frames = []
    for transcript in list_transcripts(project_dir):
        path = result_dir(project_dir, transcript) / "coded.csv"
        if path.exists():
            frames.append(read_table(path).assign(transcript=transcript.name))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[["transcript", *df.columns.drop("transcript")]]


def export_zip(project_dir: Path) -> bytes:
    """Zip the settings and results, without originals, names or findings."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project_dir.rglob("*")):
            relative = path.relative_to(project_dir)
            if (
                path.is_file()
                and path.name not in LOCAL_ONLY
                and relative.parts[0] != "transcripts"
            ):
                archive.write(path, relative.as_posix())
    return buffer.getvalue()


# --- read and split ---------------------------------------------------------


def read_transcript(path: Path) -> str:
    """Read a transcript as text. Word and PDF files are converted locally."""
    if path.suffix.lower() in {".docx", ".pdf"}:
        from docling.document_converter import DocumentConverter

        return DocumentConverter().convert(path).document.export_to_markdown()
    return path.read_text(encoding="utf-8")


def split_passages(
    text: str,
    unit: str = "turn",
    moderator_labels: list[str] | None = None,
    participant_label: str = "PARTICIPANT",
    max_chars: int = 700,
    overlap: int = 100,
) -> pd.DataFrame:
    """Split a transcript into passages, the unit that gets coded.

    - "turn": one passage per participant turn, with the moderator question
      that preceded it.
    - "question": one passage per moderator question, holding every answer.
    - "fixed": passages of `max_chars` characters, ignoring speakers.

    A speaker label can be a real name ("Rosa Méndez:"), so participants are
    renamed "<participant_label> 1", "2", ... in order of first appearance.
    A transcript without "SPEAKER:" lines is split into paragraphs.

    Returns the columns passage_id, question_id, speaker, question, text.
    """
    labels = [m.upper() for m in moderator_labels or MODERATOR_LABELS]
    turns = [] if unit == "fixed" else parse_speakers(text)
    rows: list[dict] = []
    if unit == "fixed":
        rows = [
            {"question_id": 0, "speaker": "", "question": "", "text": c.text}
            for c in chunk_text_simple(text, "", max_chars, overlap)
        ]
    elif not turns:
        rows = [
            {"question_id": 0, "speaker": "", "question": "", "text": c.text}
            for c in make_chunks(text)
        ]
    else:
        question, question_id = "", 0
        speakers: dict[str, str] = {}
        for turn in turns:
            if any(label in turn["speaker"].upper() for label in labels):
                question, question_id = turn["text"], question_id + 1
                continue
            speaker = speakers.setdefault(
                turn["speaker"], f"{participant_label} {len(speakers) + 1}"
            )
            rows.append(
                {
                    "question_id": question_id,
                    "speaker": speaker,
                    "question": question,
                    "text": turn["text"],
                }
            )
        if unit == "question":
            grouped: dict[int, dict] = {}
            for row in rows:
                line = f"{row['speaker']}: {row['text']}"
                if row["question_id"] in grouped:
                    grouped[row["question_id"]]["text"] += "\n" + line
                else:
                    grouped[row["question_id"]] = {**row, "speaker": "", "text": line}
            rows = list(grouped.values())
    df = pd.DataFrame(rows, columns=["question_id", "speaker", "question", "text"])
    df.insert(0, "passage_id", range(1, len(df) + 1))
    return df


# --- de-identify ------------------------------------------------------------


def make_cleaner(
    detector: str,
    names: list[str] | None = None,
    threshold: float = 0.5,
    keep: list[str] | None = None,
    proper_nouns_only: bool = True,
    placeholders: str = "es",
) -> Clean:
    """Build the function that replaces names, phones, emails and places.

    "presidio" works on Spanish text only; "gliner" is multilingual; "none"
    changes nothing (for text that is already de-identified). Words in `keep`
    (a country, a large city) are left in the text. `proper_nouns_only` stops
    GLiNER2-PII replacing lowercase words such as "esposo" or "casa".
    """
    if detector == "none":
        return lambda text: (text, [])
    from . import deidentify as d  # heavy imports, only when needed

    if detector == "gliner":
        model = d.load_gliner()
        return lambda text: d.deidentify_gliner(
            text,
            model,
            known_names=names,
            threshold=threshold,
            keep=keep,
            proper_nouns_only=proper_nouns_only,
            placeholders=placeholders,
        )
    analyzer = d.build_analyzer(known_names=names)
    return lambda text: d.deidentify(
        text, analyzer, keep=keep, placeholders=placeholders
    )


def deidentify_passages(
    passages: pd.DataFrame, clean: Clean
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """De-identify the text and the moderator question of every passage.

    Returns the cleaned passages and a findings table (passage_id, column,
    entity, found, score, recognizer). The findings hold the original names:
    keep them on the laptop. Detectors miss things, so read the cleaned text
    next to the original before sending it anywhere.
    """
    questions: dict[int, str] = {}
    texts: list[str] = []
    findings: list[dict] = []
    for row in passages.itertuples():
        if row.question_id not in questions:
            # The question travels with each passage, so it is cleaned too.
            cleaned, found = clean(row.question) if row.question else ("", [])
            questions[row.question_id] = cleaned
            findings += [
                {"passage_id": row.passage_id, "column": "question", **f} for f in found
            ]
        cleaned, found = clean(row.text)
        texts.append(cleaned)
        findings += [
            {"passage_id": row.passage_id, "column": "text", **f} for f in found
        ]
    out = passages.assign(question=passages["question_id"].map(questions), text=texts)
    columns = ["passage_id", "column", "entity", "found", "score", "recognizer"]
    return out, pd.DataFrame(findings, columns=columns)


# --- models -----------------------------------------------------------------


class Completer:
    """Call a model, remembering every answer on disk.

    Calling it with the same model, instructions and text a second time returns
    the saved answer, so a batch that stopped halfway resumes without paying
    again. Safe to call from several threads.
    """

    def __init__(self, client, wire: str, model: str, cache_path: Path | None = None):
        """Load the answers saved in `cache_path`, if there are any."""
        self.client, self.wire, self.model = client, wire, model
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self.calls = self.reused = self.input_tokens = self.output_tokens = 0
        self._lock = threading.Lock()
        if cache_path and cache_path.exists():
            for line in cache_path.read_text("utf-8").splitlines():
                try:
                    record = json.loads(line)
                    self.cache[record["key"]] = record["text"]
                except (ValueError, KeyError):
                    continue  # a line cut short by a crash

    def __call__(self, system: str, user: str) -> str:
        """Return the model's answer to `user` under the `system` instructions."""
        raw = json.dumps([str(self.client.base_url), self.model, system, user])
        key = hashlib.sha256(raw.encode()).hexdigest()
        with self._lock:
            if key in self.cache:
                self.reused += 1
                return self.cache[key]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Claude needs an output limit; other APIs name the setting differently
        # per model, so they keep their own default.
        extra = {"max_tokens": 16000} if self.wire == "anthropic" else {}
        text, used_in, used_out = complete_with_usage(
            self.client, self.wire, self.model, messages, **extra
        )
        with self._lock:
            self.calls += 1
            self.input_tokens += used_in
            self.output_tokens += used_out
            self.cache[key] = text
            if self.cache_path:
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "text": text}) + "\n")
        return text


def make_embedder(backend: str, model: str = "", client=None) -> Embed:
    """Build the embedding function: on the laptop, or an OpenAI-style API."""
    if backend == "local":
        from .local_embeddings import embed_texts  # loads torch, only when needed

        options = {"model_name": model} if model else {}
        return lambda texts, query=False: embed_texts(texts, query=query, **options)

    def embed(texts: list[str], query: bool = False) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):
            response = client.embeddings.create(
                model=model, input=texts[start : start + 100]
            )
            vectors += [item.embedding for item in response.data]
        return vectors

    return embed


def _pmap(fn: Callable, items: list, workers: int = 4, tick=None) -> list:
    """Run fn over items in threads, keeping order; tick(done, total) as they finish."""
    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                results[futures[future]] = future.result()
                if tick:
                    tick(done, len(items))
        except BaseException:
            pool.shutdown(cancel_futures=True)  # a bad key should not fail 500 times
            raise
    return results


def _fill(template: str, **values: str) -> str:
    """Fill {placeholders}. Unlike str.format, other braces (JSON) are left alone."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def _system(settings: dict, step: str, themes: list[Theme] = (), **extra: str) -> str:
    """Build the instructions for a step from the project's inputs."""
    study = ""
    if settings.get("research_question"):
        study += f"RESEARCH QUESTION\n{settings['research_question']}\n\n"
    if settings.get("study_context"):
        study += f"STUDY CONTEXT\n{settings['study_context']}\n\n"
    template = (settings.get("prompts") or {}).get(step) or PROMPTS[step]
    return _fill(
        template,
        study=study,
        codebook="\n\n".join(theme_prompt_text(t) for t in themes),
        off_topic=settings["off_topic"],
        output_language=settings["output_language"],
        n_themes=settings["n_themes"],
        **extra,
    )


def _passage_message(row) -> str:
    question = f"MODERATOR QUESTION\n{row.question}\n\n" if row.question else ""
    return f"{question}PASSAGE\n{row.text}"


def parse_json(text: str):
    """Read the JSON in a model answer, ignoring code fences and chatter."""
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else None
    except ValueError:
        return None


# --- coding steps -----------------------------------------------------------


def code_with_embeddings(
    passages: pd.DataFrame,
    themes: list[Theme],
    embed: Embed,
    threshold: float,
    off_topic: str,
    research_question: str = "",
) -> pd.DataFrame:
    """Code each passage with the most similar theme, without a language model.

    Adds nearest_theme, relevance (similarity to that theme) and
    embedding_theme, which is the nearest theme, or `off_topic` when the
    relevance is below `threshold`. With a research question, also adds
    question_similarity.
    """
    df = passages.copy()
    vectors = embed(df["text"].tolist())
    theme_vectors = embed([theme_prompt_text(t) for t in themes], query=True)
    themes = [
        Theme(t.short_name, t.full_definition, v)
        for t, v in zip(themes, theme_vectors, strict=True)
    ]
    names = [t.short_name for t in themes]
    # Scored in a separate table so a theme named "text" cannot overwrite a column.
    scores = pd.DataFrame({"embedding": vectors}, index=df.index)
    scores = add_theme_similarity_columns(scores, themes)[names]
    df["nearest_theme"] = classify_by_max_theme(scores, names, "t")["t"]
    df["relevance"] = scores.max(axis=1).round(2)
    df["embedding_theme"] = df["nearest_theme"].where(
        df["relevance"] >= threshold, off_topic
    )
    if research_question:
        question = embed([research_question], query=True)[0]
        df["question_similarity"] = [
            round(dot_similarity(v, question), 2) for v in vectors
        ]
    return df


def code_with_llm(
    passages: pd.DataFrame,
    themes: list[Theme],
    complete: Complete,
    settings: dict,
    prefix: str = "llm",
    tick=None,
) -> pd.DataFrame:
    """Code each passage against the codebook with a language model.

    In "single" mode each passage gets one theme (or the off-topic label) in
    `<prefix>_theme`, with a one-sentence `<prefix>_rationale`. In "multi" mode
    each theme gets its own YES/NO column, `<prefix>: <theme name>`.
    """
    df = passages.copy()
    messages = [_passage_message(row) for row in df.itertuples()]
    workers = settings["workers"]
    if settings["coding_mode"] == "multi":
        jobs = [
            (_system(settings, "code_multi", theme=theme_prompt_text(t)), m)
            for t in themes
            for m in messages
        ]
        answers = _pmap(lambda job: complete(*job), jobs, workers, tick)
        for i, theme in enumerate(themes):
            block = answers[i * len(messages) : (i + 1) * len(messages)]
            df[f"{prefix}: {theme.short_name}"] = [
                "YES" if a.strip().upper().startswith("YES") else "NO" for a in block
            ]
        return df

    system = _system(settings, "code_single", themes)
    answers = _pmap(lambda m: complete(system, m), messages, workers, tick)
    allowed = {t.short_name.casefold(): t.short_name for t in themes}
    allowed[settings["off_topic"].casefold()] = settings["off_topic"]
    coded, rationales = [], []
    for answer in answers:
        data = parse_json(answer)
        data = data if isinstance(data, dict) else {}
        theme = str(data.get("theme", "")) if data else answer
        match = allowed.get(theme.strip().casefold())
        coded.append(match or UNPARSED)
        # An unreadable answer is kept, so the researcher can see what came back.
        rationales.append(str(data.get("rationale", "")) if match else answer[:300])
    df[f"{prefix}_theme"], df[f"{prefix}_rationale"] = coded, rationales
    return df


def code_nonverbal(
    passages: pd.DataFrame, complete: Complete, settings: dict, tick=None
) -> pd.DataFrame:
    """Flag passages that record non-verbal cues: adds any_cues and cue_type."""
    df = passages.copy()
    system = _system(settings, "nonverbal")
    answers = _pmap(
        lambda text: complete(system, f"PASSAGE\n{text}"),
        df["text"].tolist(),
        settings["workers"],
        tick,
    )
    parsed = [parse_json(a) for a in answers]
    parsed = [p if isinstance(p, dict) else {} for p in parsed]
    df["any_cues"] = [
        "YES" if str(p.get("any_cues", "")).upper().startswith("Y") else "NO"
        for p in parsed
    ]
    df["cue_type"] = [str(p.get("cue_type", "")).strip() for p in parsed]
    return df


def translate(
    passages: pd.DataFrame, complete: Complete, settings: dict, tick=None
) -> pd.DataFrame:
    """Translate passages and questions; the source text moves to text_original."""
    df = passages.copy()
    system = _system(settings, "translate")
    sources = list(dict.fromkeys([*df["text"], *df["question"]]))
    sources = [s for s in sources if s.strip()]
    answers = _pmap(
        lambda text: complete(system, text), sources, settings["workers"], tick
    )
    lookup = dict(zip(sources, answers, strict=True))
    df["text_original"] = df["text"]
    df["text"] = df["text"].map(lambda t: lookup.get(t, t))
    df["question"] = df["question"].map(lambda t: lookup.get(t, t))
    return df


def discover_themes(text: str, complete: Complete, settings: dict) -> list[dict]:
    """Ask the model for the themes in a transcript (inductive coding).

    Returns dicts with name, definition and examples, the same shape as a
    codebook entry, so a proposed theme can be added to the codebook.
    """
    answer = complete(_system(settings, "themes"), f"TRANSCRIPT\n{text}")
    data = parse_json(answer)
    if isinstance(data, dict):  # some models wrap the list: {"themes": [...]}
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        return [{"name": UNPARSED, "definition": answer, "examples": ""}]
    return [
        {
            "name": str(item.get("name", "")).strip(),
            "definition": str(item.get("definition", "")).strip(),
            "examples": str(item.get("examples", "")).strip(),
        }
        for item in data
        if isinstance(item, dict) and item.get("name")
    ]


# --- compare ----------------------------------------------------------------


def agreement(a: pd.Series, b: pd.Series, a_name="human", b_name="model") -> dict:
    """Compare two sets of codes for the same passages (same index).

    Returns n, agreement (share of passages coded the same), kappa (Cohen's
    kappa, agreement beyond chance), a confusion table and the index of the
    passages coded differently.
    """
    same = float((a == b).mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # one label only: handled just below
        kappa = float(cohen_kappa_score(a, b))
    if kappa != kappa:  # undefined when both coders used a single, identical label
        kappa = 1.0
    return {
        "n": len(a),
        "agreement": same,
        "kappa": kappa,
        "confusion": pd.crosstab(a, b, rownames=[a_name], colnames=[b_name]),
        "disagreements": a.index[a != b].tolist(),
    }


def threshold_table(
    human: pd.Series, coded: pd.DataFrame, off_topic: str
) -> pd.DataFrame:
    """Agreement with the human codes for each candidate relevance threshold.

    Pick the best row, then go a little lower: a passage dropped by mistake is
    harder to notice than small talk kept by mistake.
    """
    rows = []
    for t in sorted(set(coded["relevance"])):
        model = coded["nearest_theme"].where(coded["relevance"] >= t, off_topic)
        rows.append(
            {
                "threshold": t,
                "agreement": float((human == model).mean()),
                "kappa": float(cohen_kappa_score(human, model)),
            }
        )
    return pd.DataFrame(rows)


# --- batch ------------------------------------------------------------------


def prepare(
    project_dir: Path,
    settings: dict,
    clean: Clean,
    on_progress: Progress | None = None,
    force: bool = False,
    explain: Callable[[Exception], str] = str,
    say: Callable[[str], str] = str,
) -> dict[str, str]:
    """Split and de-identify the project's new or changed transcripts.

    A transcript already prepared with the same file, roster and settings is
    left alone, with its codes, unless `force` is set. For the others this
    writes passages.csv and findings.csv and removes the codes, proposed
    themes and "reviewed" mark from an earlier run, since the passages
    changed. Returns the error message (worded by `explain`) for each
    transcript that failed; the others are still processed.
    """
    errors: dict[str, str] = {}
    transcripts = [
        t
        for t in list_transcripts(project_dir)
        if force or needs_prepare(project_dir, settings, t)
    ]
    for i, transcript in enumerate(transcripts):
        if on_progress:
            on_progress(
                i / len(transcripts),
                say("Preparing {name} ({i} of {n})").format(
                    name=transcript.name, i=i + 1, n=len(transcripts)
                ),
            )
        try:
            passages = split_passages(
                read_transcript(transcript),
                settings["unit"],
                settings["moderator_labels"],
                settings["participant_label"],
            )
            if passages.empty:
                raise ValueError(
                    say(
                        "No passages were found in this file. Check that it "
                        "holds text, and that the moderator labels on the Study "
                        "inputs page match the labels in the transcript."
                    )
                )
            passages, findings = deidentify_passages(passages, clean)
            out = result_dir(project_dir, transcript)
            out.mkdir(parents=True, exist_ok=True)
            passages.to_csv(out / "passages.csv", index=False)
            findings.to_csv(out / "findings.csv", index=False)
            for stale in ("coded.csv", "themes.json", "reviewed.txt"):
                (out / stale).unlink(missing_ok=True)
            (out / "prepared.txt").write_text(
                prep_fingerprint(project_dir, settings, transcript), "utf-8"
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            errors[transcript.name] = explain(exc)
    return errors


STEPS = {
    "translate": "translating",
    "embeddings": "similarity pass",
    "llm": "coding with the language model",
    "llm2": "coding with the second model",
    "nonverbal": "flagging non-verbal cues",
    "themes": "proposing themes",
}


def run_batch(
    project_dir: Path,
    settings: dict,
    steps: list[str],
    complete: Complete | None = None,
    embed: Embed | None = None,
    complete2: Complete | None = None,
    on_progress: Progress | None = None,
    explain: Callable[[Exception], str] = str,
    say: Callable[[str], str] = str,
) -> dict[str, str]:
    """Run the chosen steps (keys of STEPS) on every prepared transcript.

    Each transcript's coded.csv keeps the columns from steps run earlier, and
    is saved as soon as that transcript finishes. Returns the error message
    (worded by `explain`) for each transcript that failed; the others are
    still processed. Model answers already in the cache are not requested
    again.
    """
    errors: dict[str, str] = {}
    themes = themes_from_data(settings["themes"])
    steps = [step for step in STEPS if step in steps]
    todo = [
        t
        for t in list_transcripts(project_dir)
        if (result_dir(project_dir, t) / "passages.csv").exists()
    ]
    for i, transcript in enumerate(todo):
        out = result_dir(project_dir, transcript)

        def progress(step: str, transcript=transcript, i=i):
            """Report progress within one step; the bar only moves forward."""
            position = steps.index(step)

            def tick(done: int, total: int) -> None:
                if on_progress:
                    share = (position + done / max(total, 1)) / len(steps)
                    text = say("{name} ({i} of {n}): {step}").format(
                        name=transcript.name,
                        i=i + 1,
                        n=len(todo),
                        step=say(STEPS[step]),
                    )
                    if total > 1:
                        text += " " + say("({done} of {total})").format(
                            done=done, total=total
                        )
                    on_progress((i + share) / len(todo), text)

            tick(0, 1)
            return tick

        try:
            coded = out / "coded.csv"
            df = read_table(coded if coded.exists() else out / "passages.csv")
            if "translate" in steps and "text_original" not in df:
                df = translate(df, complete, settings, progress("translate"))
            if "embeddings" in steps:
                progress("embeddings")
                df = code_with_embeddings(
                    df,
                    themes,
                    embed,
                    settings["threshold"],
                    settings["off_topic"],
                    settings["research_question"],
                )
            if "llm" in steps:
                tick = progress("llm")
                df = code_with_llm(df, themes, complete, settings, "llm", tick)
            if "llm2" in steps:
                tick = progress("llm2")
                df = code_with_llm(df, themes, complete2, settings, "llm2", tick)
            if "nonverbal" in steps:
                df = code_nonverbal(df, complete, settings, progress("nonverbal"))
            df.to_csv(coded, index=False)
            if "themes" in steps:
                progress("themes")
                text = "\n".join(
                    f"{row.speaker}: {row.text}" if row.speaker else row.text
                    for row in df.itertuples()
                )
                (out / "themes.json").write_text(
                    json.dumps(
                        discover_themes(text, complete, settings),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            errors[transcript.name] = explain(exc)
    if on_progress:
        on_progress(1.0, say("Done"))
    return errors
