"""Web app for the qualitative coding pipeline.

Launch with `just app` (or `uv run streamlit run app/main.py`).

The app is only the screens. Every analysis step lives in src/pipeline.py, so
the same functions can be used from a script or notebook: do not add analysis
logic here.

The look follows DataSure, IPA's data quality tool, so the two read as one
family: the same canvas, text colour, font, page header, bordered metric rows,
confirmation dialog and sidebar footer. Colour carries one meaning here that
DataSure does not need: green means "stays on this computer". Buttons,
headings and ordinary confirmations never use it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from i18n import (  # noqa: E402
    N_,
    Tr,
    language,
    languages,
    remember,
    start,
    tr,
)

from src import pipeline as pl  # noqa: E402
from src.coding import (  # noqa: E402
    dump_themes,
    load_themes,
    themes_from_data,
    themes_to_data,
)
from src.llm_client import LOCAL_BASE_URL, make_client  # noqa: E402

load_dotenv(ROOT / ".env")

APP_NAME = "Qualitative Coding"
ASSETS = Path(__file__).resolve().parent / "assets"
SUPPORT_EMAIL = "support@poverty-action.org"
DOCS_URL = "https://github.com/PovertyAction/llm-quali-coding#use-the-web-app"
ISSUES_URL = "https://github.com/PovertyAction/llm-quali-coding/issues"

# The parts of the page Streamlit draws with its own defaults, brought up to
# readable contrast on the grey canvas: captions, placeholders, menu links
# and a focus outline a keyboard user can see. Running text is capped near 75
# characters a line. The edges of inputs and tick boxes are set in the theme
# (borderColor in .streamlit/config.toml).
CSS = """<style>
[data-testid="stCaptionContainer"] { opacity: 1; color: #505052; }
code { color: #414042 !important; }
input::placeholder, textarea::placeholder { color: #505052 !important; opacity: 1;
  font-style: italic; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li { max-width: 75ch; }
h2 { font-size: 1.75rem !important; }
[data-testid="stSidebarNavLink"] span { color: #414042 !important; }
:focus-visible { outline: 2px solid #c8420a !important; outline-offset: 2px; }
[data-testid="stCheckbox"] label:has(input:focus-visible),
[data-testid="stRadio"] label:has(input:focus-visible) { outline: 2px solid #c8420a;
  outline-offset: 2px; border-radius: 4px; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
::selection { background: #f5cb57; color: #414042; }
.stMainBlockContainer { padding-top: 3rem; }
</style>"""

BACKENDS = Tr(
    {
        "local": N_("On this computer: Ollama or LM Studio"),
        "anthropic": N_("Cloud API: Anthropic (Claude)"),
        "openai": N_("Cloud API: OpenAI"),
        "azure": N_("Institution cloud: Azure OpenAI"),
        "bedrock": N_("Institution cloud: AWS Bedrock (Claude)"),
    }
)
# The highest IPA data classification each service is approved for. None means
# no approval is recorded here. Change this table when the approvals change.
APPROVED = {
    "local": "Confidential",
    "azure": "Confidential",
    "anthropic": "Internal",
    "bedrock": "Internal",
    "openai": None,
}
LEVELS = Tr(
    {
        "Confidential": N_("Public, Internal or Confidential"),
        "Internal": N_("Public or Internal"),
    }
)
# Marked for translation; shown through tr() in approval_short().
LEVEL_NAMES = (N_("Confidential"), N_("Internal"))
RANK = {None: 0, "Internal": 1, "Confidential": 2}
DEFAULT_MODELS = {
    "local": "",
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-5-mini",
    "azure": "",
    "bedrock": "anthropic.claude-opus-5-5",
}
KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
}
EMBED_BACKENDS = Tr(
    {
        "local": N_("On this computer (open-weights model)"),
        "openai": N_("Cloud API: OpenAI"),
        "azure": N_("Institution cloud: Azure OpenAI"),
    }
)
UNITS = Tr(
    {
        "turn": N_("Each participant turn"),
        "question": N_("All answers to one moderator question"),
        "fixed": N_("Fixed-size chunks (ignores speakers)"),
    }
)
DETECTORS = Tr(
    {
        "gliner": N_("GLiNER2-PII (many languages)"),
        "presidio": N_("Presidio (Spanish only)"),
        "none": N_("None: the transcripts are already de-identified"),
    }
)
# Steps on the Run page, in the order they are shown.
STEP_LABELS = Tr(
    {
        "embeddings": N_("Similarity pass: give each passage its most similar theme"),
        "llm": N_("Code each passage against the codebook"),
        "llm2": N_("Code again with a second model, to compare the two"),
        "nonverbal": N_("Flag non-verbal cues such as laughter or pauses"),
        "themes": N_("Propose new themes from each transcript"),
        "translate": N_("Translate the passages first"),
    }
)
STEP_NAMES = Tr(
    {
        "embeddings": N_("similarity pass"),
        "llm": N_("language model coding"),
        "llm2": N_("second model coding"),
        "nonverbal": N_("non-verbal cues"),
        "themes": N_("theme proposals"),
        "translate": N_("translation"),
    }
)
CODE_COLUMNS = Tr(
    {
        "embedding_theme": N_("Similarity pass"),
        "llm_theme": N_("Language model"),
        "llm2_theme": N_("Second model"),
    }
)
COLUMN_LABELS = Tr(
    {
        "transcript": N_("Transcript"),
        "passage_id": N_("Passage"),
        "speaker": N_("Speaker"),
        "question": N_("Moderator question"),
        "text": N_("Text"),
        "text_original": N_("Text before translation"),
        "embedding_theme": N_("Theme (similarity pass)"),
        "nearest_theme": N_("Nearest theme"),
        "relevance": N_("Relevance"),
        "question_similarity": N_("Similarity to research question"),
        "llm_theme": N_("Theme (language model)"),
        "llm_rationale": N_("Reason (language model)"),
        "llm2_theme": N_("Theme (second model)"),
        "llm2_rationale": N_("Reason (second model)"),
        "any_cues": N_("Non-verbal cue"),
        "cue_type": N_("Cue"),
    }
)
THEME_FIELDS = ["name", "definition", "include", "exclude", "examples"]
NOT_CHAT = re.compile(r"embed|bge|minilm|rerank|ocr", re.IGNORECASE)
# Proper names of the services, the same in every language.
SERVICES = {
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "bedrock": "AWS Bedrock (Claude)",
}
# Nouns that are counted: singular and plural.
NOUNS = {
    "transcript": (N_("transcript"), N_("transcripts")),
    "changed": (N_("new or changed transcript"), N_("new or changed transcripts")),
    "theme": (N_("theme"), N_("themes")),
    "passage": (N_("passage"), N_("passages")),
    "item": (N_("item"), N_("items")),
    "request": (N_("request"), N_("requests")),
    "file": (N_("file"), N_("files")),
    "row": (N_("row"), N_("rows")),
}
# A sample of the tags each placeholder language uses.
TAGS = {"en": "<PERSON>", "es": "<PERSONA>", "fr": "<PERSONNE>"}
LANGUAGE_LABEL = "Language · Idioma · Langue"
PING = ("You are a connection test.", "Reply with the single word OK.")
# Invented focus groups in data/fictitious, one per language.
EXAMPLES = {
    "en": {
        "project": "example_en",
        "files": ("focus_group_fictitious_en.md", "participant_list_en.txt"),
        "codebook": "codebook_fictitious_en.json",
        "human": "human_codes_en.csv",
        "settings": {
            "research_question": "What helps or hinders members in saving with "
            "the group?",
            "off_topic": "Off topic",
            "participant_label": "PARTICIPANT",
            "moderator_labels": ["MODERATOR"],
            "detector": "gliner",
            "placeholders": "en",
            "output_language": "English",
        },
    },
    "es": {
        "project": "example",
        "files": ("grupo_focal_ficticio.md", "lista_participantes.txt"),
        "codebook": "codebook_ficticio.json",
        "human": "codigos_humanos.csv",
        "settings": {
            "research_question": "¿Qué facilita o dificulta que las socias "
            "ahorren en el grupo?",
            "off_topic": "Fuera de tema",
            "participant_label": "PARTICIPANTE",
            "moderator_labels": ["MODERADOR"],
            "detector": "presidio",
            "placeholders": "es",
            "output_language": "Spanish",
        },
    },
    "fr": {
        "project": "example_fr",
        "files": ("groupe_de_discussion_fictif.md", "liste_participants.txt"),
        "codebook": "codebook_fictif.json",
        "human": "codes_humains.csv",
        "settings": {
            "research_question": "Qu'est-ce qui aide ou freine l'épargne des "
            "membres au sein du groupe ?",
            "off_topic": "Hors sujet",
            "participant_label": "PARTICIPANT",
            "moderator_labels": ["ANIMATRICE", "ANIMATEUR"],
            "detector": "gliner",
            "placeholders": "fr",
            "output_language": "French",
        },
    },
}
PER_PAGE = 10

# Filled in by main(), so a page can link to the next step. Empty when a page
# function is called on its own, as the tests do.
PAGES: dict[str, object] = {}


# --- shared UI helpers (the same idioms as DataSure's ui_utils) --------------


def page_header(title: str, subtitle: str) -> None:
    """Page title that matches the menu label, one line on what the page is for."""
    st.title(title, anchor=False)
    st.markdown(subtitle)
    st.divider()
    for level, message in st.session_state.pop("_notices", []):
        if level == "toast":
            st.toast(message, icon=":material/check:")
        else:
            getattr(st, level)(message)


def notice(level: str, message: str) -> None:
    """Queue a message for the next run; one drawn just before a rerun is lost."""
    st.session_state.setdefault("_notices", []).append((level, message))


def count(n: int, noun: str) -> str:
    """Format a number with its noun (a key of NOUNS): "2 transcripts"."""
    one, many = NOUNS[noun]
    return f"{n:,} {tr(one) if n == 1 else tr(many)}"


def metric_row(metrics: list[tuple]) -> None:
    """Row of equal, bordered metrics: (label, value) or (label, value, help)."""
    columns = st.columns(len(metrics), border=True)
    for column, metric in zip(columns, metrics, strict=True):
        column.metric(metric[0], metric[1], help=metric[2] if len(metric) > 2 else None)


def confirm_dialog(
    title: str, body: str, confirm_label: str, on_confirm: Callable[[], None]
) -> None:
    """Ask before an action that deletes work. The only confirmation idiom.

    `on_confirm` should be quick: slow work is flagged in session state and
    done by the page after the dialog closes, where a click cannot cancel it
    halfway.
    """

    @st.dialog(title)
    def dialog() -> None:
        st.warning(body, icon=":material/warning:")
        no, yes = st.columns(2)
        if no.button(tr("Cancel"), width="stretch"):
            st.rerun()
        if yes.button(confirm_label, type="primary", width="stretch"):
            on_confirm()
            st.rerun()

    dialog()


def next_step(page: str, label: str) -> None:
    """Point to the next page once this one's work is done."""
    st.divider()
    if page in PAGES:
        st.page_link(
            PAGES[page],
            label=tr("**Next: {label}**").format(label=label),
            icon=":material/arrow_forward:",
        )
    else:
        st.markdown(tr("**Next: {label}**").format(label=label))


def explain(exc: Exception) -> str:
    """Turn a model or file error into a sentence that says what to do."""
    hints = {
        "AuthenticationError": tr(
            "The service did not accept the API key. Check the "
            "key on the Project and models page."
        ),
        "PermissionDeniedError": tr(
            "This key is not allowed to use that model. "
            "Choose another model on the Project and models page, or ask whoever "
            "issued the key to enable it."
        ),
        "APIConnectionError": tr(
            "The model could not be reached. If it runs on "
            "this computer, check that Ollama or LM Studio is open. Otherwise check "
            "the internet connection and the server address."
        ),
        "APITimeoutError": tr(
            "The model took too long to answer. Run again: the "
            "answers already received are kept."
        ),
        "NotFoundError": tr(
            "The service does not know this model name. Check the "
            "spelling on the Project and models page."
        ),
        "RateLimitError": tr(
            "The service is limiting requests. Wait a minute and "
            "run again, or lower the number of requests sent at the same time "
            "(Project and models, under Advanced)."
        ),
        "ValueError": str(exc),
    }
    return hints.get(type(exc).__name__) or tr("Something went wrong: {exc}").format(
        exc=exc
    )


def show_errors(errors: dict[str, str], verb: str) -> None:
    """Show each distinct problem once, with the transcripts it happened to."""
    grouped: dict[str, list[str]] = {}
    for name, message in errors.items():
        grouped.setdefault(message, []).append(name)
    for message, names in grouped.items():
        who = names[0] if len(names) == 1 else count(len(names), "transcript")
        st.error(f"**{who} {verb}.** {message}", icon=":material/error:")
        if len(names) > 1:
            st.caption(", ".join(names))


def md_escape(text: str) -> str:
    """Show transcript text literally: stop Markdown reading its punctuation."""
    text = re.sub(r"([\\`*_{}\[\]<>#$~|:])", r"\\\1", text)
    return re.sub(r"^(\s*)([-+]|\d+[.)])(\s)", r"\1\\\2\3", text, flags=re.MULTILINE)


def highlight_placeholders(text: str) -> str:
    """Escape de-identified text and mark the placeholders the detector put in."""
    return re.sub(
        r"\\<([A-ZÁÉÍÓÚÑ\\_]{3,})\\>", r":blue-background[\\<\1\\>]", md_escape(text)
    )


def page_rows(total: int, key: str) -> range:
    """Rows of a long list to show now; pager() below the list changes the page."""
    pages = max(1, -(-total // PER_PAGE))
    if st.session_state.get(f"page_{key}", 1) > pages:
        st.session_state[f"page_{key}"] = 1
    page = st.session_state.get(f"page_{key}", 1)
    return range((page - 1) * PER_PAGE, min(page * PER_PAGE, total))


def pager(total: int, key: str) -> None:
    """Page control under a long list, saying which rows are on screen."""
    pages = max(1, -(-total // PER_PAGE))
    if pages > 1:
        control, shown = st.columns([1, 3], vertical_alignment="bottom")
        page = control.number_input(
            tr("Page (of {pages})").format(pages=pages), 1, pages, key=f"page_{key}"
        )
        first = (page - 1) * PER_PAGE + 1
        shown.caption(
            tr("Showing {first} to {x1} of {total:,}").format(
                first=first, x1=min(page * PER_PAGE, total), total=total
            )
        )


# --- project helpers --------------------------------------------------------


def projects_dir() -> Path:
    """Folder holding every project (git-ignored)."""
    return Path(os.environ.get("QUALI_PROJECTS_DIR", ROOT / "outputs" / "projects"))


def open_project(name: str) -> None:
    """Switch to a project on the next run (the picker cannot be set mid-run)."""
    st.session_state["_open"] = name
    st.rerun()


def current() -> tuple[Path, dict]:
    """Folder and settings of the open project; stops the page if there is none."""
    name = st.session_state.get("project")
    if not name or not (projects_dir() / name).exists():
        st.info(
            tr(
                "**No project is open.** Create one, or open the example, on the "
                "**Project and models** page."
            ),
            icon=":material/folder_open:",
        )
        st.stop()
    folder = projects_dir() / name
    return folder, pl.load_settings(folder)


def save_if_changed(folder: Path, old: dict, new: dict) -> None:
    """Save the settings when a widget changed them, say so, and redraw.

    The menu check marks, the sidebar's statement of where the text goes and
    the tables on the page are all computed from the saved settings, so the
    page is run again: nothing on screen shows the previous answer.
    """
    if new != old:
        pl.save_settings(folder, new)
        notice("toast", tr("Saved"))
        st.rerun()


def server_host(settings: dict) -> str:
    """Host of the "local" model server."""
    return urlparse(settings["base_url"] or LOCAL_BASE_URL).hostname or ""


def cloud_destinations(
    settings: dict, steps: list[str] | None = None
) -> list[tuple[str, str]]:
    """Outside services that would receive text, as (approval key, name)."""
    steps = list(pl.STEPS) if steps is None else steps
    found: dict[str, str] = {}
    if any(step != "embeddings" for step in steps):
        backend = settings["backend"]
        if backend != "local":
            found[backend] = SERVICES[backend]
        elif server_host(settings) not in {"localhost", "127.0.0.1", "::1"}:
            # A "local" server on another machine is not this computer.
            found["remote"] = tr("the model server at {x0}").format(
                x0=server_host(settings)
            )
    if "embeddings" in steps and settings["embed_backend"] != "local":
        key = settings["embed_backend"]
        found[key] = SERVICES[key]
    return list(found.items())


def approval(destinations: list[tuple[str, str]]) -> str | None:
    """Highest data classification every destination is approved for, if any."""
    levels = [APPROVED.get(key) for key, _ in destinations]
    return min(levels, key=RANK.get) if levels else "Confidential"


def approval_note(key: str) -> str:
    """One sentence on what a service is approved for."""
    level = APPROVED.get(key)
    if level:
        return tr("Approved for {x0} data.").format(x0=LEVELS[level])
    return tr("No approval is recorded for this service.")


def backend_label(key: str) -> str:
    """Model location with its approval, for the picker."""
    return f"{BACKENDS[key]} · {approval_short(key)}"


def approval_short(key: str) -> str:
    """Approval in a few words, for a list."""
    level = APPROVED.get(key)
    if not level:
        return tr("no approval recorded")
    return tr("up to {level}").format(level=tr(level))


def client_for(settings: dict, backend: str):
    """Build a client from the project settings and the keys typed this session."""
    return make_client(
        backend,
        api_key=st.session_state.get("keys", {}).get(backend) or None,
        base_url=settings["base_url"],
        azure_endpoint=settings["azure_endpoint"],
        api_version=settings["api_version"],
        aws_region=settings["aws_region"],
        aws_profile=settings["aws_profile"],
    )


@st.cache_data(ttl=30, show_spinner=tr("Looking for Ollama or LM Studio..."))
def local_models(base_url: str) -> list[str]:
    """Models installed on the local server; empty when it is not running."""
    try:
        client, _ = make_client("local", base_url=base_url, max_retries=0)
        return sorted(m.id for m in client.with_options(timeout=2).models.list())
    except Exception:  # noqa: BLE001 - not running is a normal state
        return []


@st.cache_resource(show_spinner=tr("Loading the de-identification model..."))
def cached_cleaner(
    detector: str,
    names: tuple[str, ...],
    threshold: float,
    keep: tuple[str, ...],
    proper_nouns_only: bool,
    placeholders: str,
):
    """Keep the detector loaded between runs."""
    return pl.make_cleaner(
        detector, list(names), threshold, list(keep), proper_nouns_only, placeholders
    )


@st.cache_data(show_spinner=tr("Reading the transcript..."))
def original_passages(path: str, modified: float, unit: str, labels: tuple, who: str):
    """Passages before de-identification, for the side-by-side review."""
    return pl.split_passages(pl.read_transcript(Path(path)), unit, list(labels), who)


def read_roster(folder: Path) -> list[str]:
    """Participant names for the de-identification step."""
    path = folder / "roster.txt"
    lines = path.read_text("utf-8").splitlines() if path.exists() else []
    return [line.strip() for line in lines if line.strip()]


def last_run(folder: Path) -> dict | None:
    """Record of the project's most recent run, if it has one."""
    path = folder / "last_run.json"
    try:
        return json.loads(path.read_text("utf-8")) if path.exists() else None
    except ValueError:
        return None


def codes_outdated(folder: Path, settings: dict) -> bool:
    """Whether the codebook or coding settings changed after the last run."""
    record = last_run(folder)
    return bool(
        record
        and record.get("state") == "done"
        and record.get("fingerprint")
        != pl.coding_fingerprint(settings, record.get("steps"))
    )


def outdated_warning() -> None:
    """Say that the codes on screen predate the current codebook or settings."""
    st.warning(
        tr(
            "**The codebook, model or coding settings changed after the last run.** "
            "The codes shown were made before that change. Run again on the **Run** "
            "page to bring them up to date; answers that are still valid are reused."
        ),
        icon=":material/history:",
    )


def make_example(code: str) -> None:
    """Create a project from one of the invented focus groups in data/fictitious."""
    example = EXAMPLES[code]
    folder = projects_dir() / example["project"]
    data = ROOT / "data" / "fictitious"
    if not (folder / "project.json").exists():
        transcript, roster = example["files"]
        (folder / "transcripts").mkdir(parents=True, exist_ok=True)
        shutil.copy(data / transcript, folder / "transcripts")
        shutil.copy(data / roster, folder / "roster.txt")
        pl.save_settings(
            folder,
            {
                **pl.DEFAULTS,
                **example["settings"],
                "themes": themes_to_data(load_themes(data / example["codebook"])),
                "steps": ["embeddings"],
            },
        )


def new_project_settings() -> dict:
    """Defaults for a new project, in the interface language."""
    code = language()
    return {
        **pl.DEFAULTS,
        "off_topic": tr("Off topic"),
        "participant_label": tr("PARTICIPANT"),
        "output_language": languages().get(code, "English"),
        "placeholders": code if code in TAGS else "en",
    }


# --- pages ------------------------------------------------------------------


def project_page() -> None:
    """Create or open a project and choose the models."""
    page_header(
        tr("Project and models"),
        tr(
            "Code interview and focus group transcripts against your codebook, on "
            "this computer or with a cloud model."
        ),
    )
    has_project = bool(st.session_state.get("project"))
    with st.container(border=True):
        if not has_project:
            st.markdown(
                tr(
                    "**Start with an example.** Each is an invented focus group with "
                    "a codebook and a roster, in English, Spanish or French. The "
                    "first pass needs no API key and no extra software."
                )
            )
        example, new = st.columns([1, 2], vertical_alignment="bottom")
        with example:
            codes = list(EXAMPLES)
            which = st.selectbox(
                tr("Example transcript in"),
                codes,
                index=codes.index(language()) if language() in codes else 0,
                format_func=lambda code: languages().get(code, code),
            )
            if st.button(
                tr("Open the example project"),
                type="secondary" if has_project else "primary",
                icon=":material/science:",
            ):
                make_example(which)
                notice(
                    "info",
                    tr(
                        "**The example project is open.** Work down the menu on the "
                        "left: look at the codebook, prepare the transcript, then run "
                        "the similarity pass."
                    ),
                )
                open_project(EXAMPLES[which]["project"])
        with new, st.form("new_project", border=False):
            name_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
            name = name_col.text_input(tr("New project name"))
            if button_col.form_submit_button(tr("Create project")):
                slug = re.sub(r"[^\w-]+", "_", name.strip()).strip("_")
                if not slug:
                    st.error(
                        tr("Type a name for the project, using letters or numbers.")
                    )
                elif (projects_dir() / slug / "project.json").exists():
                    st.error(
                        tr(
                            "A project named **{slug}** already exists. Pick it in the Project "
                            "list in the left menu, or choose another name."
                        ).format(slug=slug)
                    )
                else:
                    pl.save_settings(projects_dir() / slug, new_project_settings())
                    notice(
                        "info",
                        tr("**Project {slug} created.** Choose a model below.").format(
                            slug=slug
                        ),
                    )
                    open_project(slug)
    st.caption(
        tr(
            "A project keeps your study inputs, transcripts and results in one "
            "folder on this computer: `{x0}`"
        ).format(x0=projects_dir())
    )
    if not has_project:
        st.stop()

    folder, old = current()
    new = dict(old)
    st.header(tr("Language model"), anchor=False)
    st.markdown(
        tr(
            "The language model reads each passage and chooses a theme. Each place "
            "it can run is approved for data up to a classification, shown in the "
            "list. The similarity pass on the Run page works without one."
        )
    )
    backend = st.selectbox(
        tr("Where the language model runs"),
        list(BACKENDS),
        index=list(BACKENDS).index(old["backend"]),
        format_func=backend_label,
    )
    new["backend"] = backend
    if backend == "local":
        st.success(
            tr("Text stays on this computer. {x0}").format(x0=approval_note(backend)),
            icon=":material/computer:",
        )
    elif APPROVED.get(backend):
        st.warning(
            tr("Passages are sent to {x0}. {x1}").format(
                x0=SERVICES[backend], x1=approval_note(backend)
            ),
            icon=":material/cloud_upload:",
        )
    else:
        st.error(
            tr(
                "Passages are sent to {x0}. {x1} Use it only for data you are "
                "allowed to send there."
            ).format(x0=SERVICES[backend], x1=approval_note(backend)),
            icon=":material/cloud_upload:",
        )
    models = dict(old["models"])
    if old["model"]:
        models.setdefault(old["backend"], old["model"])
    saved_model = models.get(backend, "")
    keys = st.session_state.setdefault("keys", {})
    if backend in KEY_ENV:
        keys[backend] = st.text_input(
            tr("API key"),
            value=keys.get(backend, os.environ.get(KEY_ENV[backend], "")),
            type="password",
            help=tr(
                "Kept for this session only and never saved. Defaults to {x0} in "
                "the .env file."
            ).format(x0=KEY_ENV[backend]),
        )
    if backend == "local":
        found = local_models(old["base_url"])
        if found:
            chat = [m for m in found if not NOT_CHAT.search(m)] or found
            options = found if saved_model in found else [*found, saved_model]
            options = [m for m in options if m]
            new["model"] = st.selectbox(
                tr("Model"),
                options,
                index=options.index(saved_model or chat[0]),
                help=tr(
                    "Pick a chat model. Models with 'embed' or 'bge' in the "
                    "name cannot code passages."
                ),
            )
        else:
            st.info(
                tr(
                    "**No model server was found on this computer.** To code with a "
                    "language model here, install [Ollama](https://ollama.com) or "
                    "[LM Studio](https://lmstudio.ai), open it, download a model in "
                    "its window, then reload this page. You can do the similarity "
                    "pass without it."
                ),
                icon=":material/dns:",
            )
            new["model"] = st.text_input(tr("Model name"), value=saved_model)
        with st.expander(tr("Model server address")):
            new["base_url"] = st.text_input(
                tr("Address"),
                value=old["base_url"],
                placeholder=LOCAL_BASE_URL,
                help=tr(
                    "Leave empty for Ollama on this computer. LM Studio: "
                    "http://localhost:1234/v1."
                ),
            )
    else:
        if backend == "azure":
            new["azure_endpoint"] = st.text_input(
                tr("Azure endpoint"),
                value=old["azure_endpoint"],
                placeholder="https://my-resource.openai.azure.com",
            )
            new["api_version"] = st.text_input(
                tr("API version"), value=old["api_version"] or "2024-10-21"
            )
        if backend == "bedrock":
            new["aws_region"] = st.text_input(
                tr("AWS region"), value=old["aws_region"] or "us-east-1"
            )
            new["aws_profile"] = st.text_input(
                tr("AWS profile"),
                value=old["aws_profile"],
                help=tr(
                    "A profile from `aws configure sso` or `aws configure`. "
                    "Leave empty to use the default AWS credentials."
                ),
            )
        new["model"] = st.text_input(
            tr("Deployment name") if backend == "azure" else tr("Model"),
            value=saved_model or DEFAULT_MODELS[backend],
            help=tr(
                "Larger models code more carefully and cost more per passage. "
                "Check a sample against human codes before choosing a cheaper one."
            ),
        )
    new["models"] = {**models, backend: new["model"]} if new["model"] else models
    if st.button(tr("Test connection"), icon=":material/network_check:"):
        try:
            client, wire = client_for(new, backend)
            answer = pl.Completer(client, wire, new["model"])(*PING)
            st.info(
                tr("**Connected.** The model answered: {x0}").format(
                    x0=answer.strip()[:80]
                ),
                icon=":material/check:",
            )
        except Exception as exc:  # noqa: BLE001 - show any failure to the user
            st.error(explain(exc), icon=":material/error:")

    with st.expander(tr("Advanced: similarity model and speed")):
        st.markdown(
            tr(
                "The similarity pass compares each passage with each theme using an "
                "embedding model, which turns text into numbers so that similar "
                "meanings score close together. The default runs on this computer "
                "and downloads about 210 MB the first time it is used."
            )
        )
        new["embed_backend"] = st.selectbox(
            tr("Where the embedding model runs"),
            list(EMBED_BACKENDS),
            index=list(EMBED_BACKENDS).index(old["embed_backend"]),
            format_func=lambda k: f"{EMBED_BACKENDS[k]} · {approval_short(k)}",
        )
        if new["embed_backend"] != backend and new["embed_backend"] in KEY_ENV:
            env = KEY_ENV[new["embed_backend"]]
            keys[new["embed_backend"]] = st.text_input(
                tr("Embedding API key"),
                value=keys.get(new["embed_backend"], os.environ.get(env, "")),
                type="password",
            )
        new["embed_model"] = st.text_input(
            tr("Embedding model"),
            value=old["embed_model"],
            placeholder=tr("Default")
            if new["embed_backend"] == "local"
            else "text-embedding-3-small",
            help=tr(
                "Leave empty for the default. Every embedding model scores on "
                "its own scale: after changing it, set the relevance threshold again."
            ),
        )
        new["workers"] = st.number_input(
            tr("Language model requests sent at the same time"),
            1,
            16,
            int(old["workers"]),
            help=tr(
                "Higher is faster on cloud models. A model on this computer "
                "answers one request at a time whatever this says."
            ),
        )
    save_if_changed(folder, old, new)

    with st.expander(tr("Delete this project")):
        st.markdown(
            tr(
                "Deletes the folder `{x0}`: its settings, its copies of the "
                "transcripts and every result. Your own transcript files are not "
                "touched."
            ).format(x0=folder.name)
        )
        if st.button(tr("Delete project"), icon=":material/delete:"):

            def delete() -> None:
                shutil.rmtree(folder)
                st.session_state.pop("project", None)
                notice(
                    "info", tr("**Project {x0} was deleted.**").format(x0=folder.name)
                )

            confirm_dialog(
                tr("Delete project"),
                tr(
                    "This deletes the project **{x0}** and all its results. This "
                    "cannot be undone."
                ).format(x0=folder.name),
                tr("Delete"),
                delete,
            )
    next_step("study", tr("write the research question and the codebook"))


def study_page() -> None:
    """Research question, codebook and the other qualitative inputs."""
    page_header(
        tr("Study inputs"),
        tr(
            "What you are asking, the themes you code for, and how the text is "
            "split up. Changes are saved when you leave a field."
        ),
    )
    folder, old = current()
    new = dict(old)
    new["research_question"] = st.text_area(
        tr("Research question"), old["research_question"], height=80
    )
    new["study_context"] = st.text_area(
        tr("Study context (no names or other identifying details)"),
        old["study_context"],
        height=100,
        help=tr(
            "Who was interviewed, about which program, and anything a new "
            "coder would need to know. Given to the language model with every "
            "passage."
        ),
    )

    st.header(tr("Codebook"), anchor=False)
    st.markdown(
        tr(
            "One row per theme. Give each a name and a definition. *Include when*, "
            "*Exclude when* and *Example quotes* are optional guidance, read by the "
            "language model. To delete a row, tick it and press Delete."
        )
    )
    table = pd.DataFrame(old["themes"], columns=THEME_FIELDS).fillna("")
    edited = st.data_editor(
        table,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn(tr("Name"), width="medium"),
            "definition": st.column_config.TextColumn(tr("Definition"), width="large"),
            "include": st.column_config.TextColumn(tr("Include when"), width="medium"),
            "exclude": st.column_config.TextColumn(tr("Exclude when"), width="medium"),
            "examples": st.column_config.TextColumn(
                tr("Example quotes"), width="medium"
            ),
        },
    )
    themes = themes_from_data(edited.fillna("").astype(str).to_dict("records"))
    new["themes"] = themes_to_data(themes)
    names = [t.short_name for t in themes]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        st.warning(
            tr(
                "**More than one theme is named {x0}.** Give each theme its own "
                "name, or the codes cannot be told apart."
            ).format(x0=", ".join(repeated))
        )

    with st.expander(tr("Load or download a codebook file")):
        upload = st.file_uploader(
            tr("Codebook file (JSON or CSV)"),
            type=["json", "csv"],
            help=tr(
                'JSON: a list of "Name: definition" strings, or of objects '
                "with name, definition, include, exclude, examples. CSV: the same "
                "column names."
            ),
        )
        if upload and st.button(tr("Load this codebook")):
            try:
                if upload.name.lower().endswith(".json"):
                    data = json.loads(upload.getvalue().decode("utf-8"))
                else:
                    data = pd.read_csv(upload).fillna("").to_dict("records")
                loaded = themes_to_data(themes_from_data(data))
            except Exception:  # noqa: BLE001 - any unreadable file
                loaded = []
            if not loaded:
                st.error(
                    tr(
                        "**No themes were found in this file.** A CSV needs the "
                        "columns `name` and `definition`. A JSON file needs a list "
                        'of "Name: definition" strings.'
                    )
                )
            else:

                def replace() -> None:
                    pl.save_settings(folder, {**new, "themes": loaded})
                    notice(
                        "info",
                        tr("**Loaded {x0}.**").format(x0=count(len(loaded), "theme")),
                    )

                if themes:
                    confirm_dialog(
                        tr("Replace the codebook"),
                        tr(
                            "This replaces the {x0} in the table with the {x1} in the file."
                        ).format(x0=count(len(themes), "theme"), x1=len(loaded)),
                        tr("Replace"),
                        replace,
                    )
                else:
                    replace()
                    st.rerun()
        left, right = st.columns(2)
        left.download_button(
            tr("Download as JSON"),
            dump_themes(themes),
            "codebook.json",
            "application/json",
        )
        right.download_button(
            tr("Download as CSV"),
            pd.DataFrame(new["themes"], columns=THEME_FIELDS).to_csv(index=False),
            "codebook.csv",
            "text/csv",
        )

    st.header(tr("Passages"), anchor=False)
    st.markdown(
        tr(
            "A passage is the piece of text that gets a code. Changing anything in "
            "this section means preparing the transcripts again."
        )
    )
    with st.container(border=True):
        new["unit"] = st.selectbox(
            tr("What counts as a passage"),
            list(UNITS),
            index=list(UNITS).index(old["unit"]),
            format_func=UNITS.get,
        )
        left, right = st.columns(2)
        labels = left.text_input(
            tr("Moderator labels"),
            ", ".join(old["moderator_labels"]),
            help=tr(
                "Speaker labels that mark the moderator or interviewer, "
                "separated by commas. Their turns become the question attached to "
                "each passage and are not coded."
            ),
        )
        new["moderator_labels"] = [x.strip() for x in labels.split(",") if x.strip()]
        new["participant_label"] = right.text_input(
            tr("Participant label"),
            old["participant_label"],
            help=tr(
                'Speakers are renamed "<label> 1", "<label> 2", ... because a '
                "speaker label can be a real name."
            ),
        )

    st.header(tr("Coding"), anchor=False)
    with st.container(border=True):
        new["coding_mode"] = st.radio(
            tr("How the language model codes a passage"),
            ["single", "multi"],
            index=["single", "multi"].index(old["coding_mode"]),
            format_func={
                "single": tr("One theme per passage, with a one-sentence reason"),
                "multi": tr("Yes or no for every theme (a passage can have several)"),
            }.get,
            help=tr(
                "Yes or no for every theme sends one request per passage per "
                "theme, so it costs more."
            ),
        )
        left, right = st.columns(2)
        new["off_topic"] = left.text_input(
            tr("Label for passages that fit no theme"), old["off_topic"]
        )
        new["output_language"] = right.text_input(
            tr("Language for reasons, proposed themes and translation"),
            old["output_language"],
        )

    with st.expander(tr("Advanced: similarity pass and theme proposals")):
        new["threshold"] = st.slider(
            tr("Relevance threshold for the similarity pass"),
            0.0,
            1.0,
            float(old["threshold"]),
            0.01,
            help=tr(
                "Each passage gets a relevance score: how similar it is to its "
                "nearest theme, from 0 to 1. Passages below the threshold get the "
                "no-theme label."
            ),
        )
        st.markdown(
            tr(
                "There is no right value in general. Code a sample by hand, then "
                "**Check against human codes** shows the agreement for each value."
            )
        )
        new["n_themes"] = st.text_input(
            tr("How many themes to propose per transcript"),
            old["n_themes"],
            help=tr("A number or a range, for example 8-15."),
        )

    with st.expander(tr("Advanced: instructions sent to the language model")):
        st.markdown(
            tr(
                "Words in {braces} are filled in from the inputs above: `{study}` "
                "(research question and context), `{codebook}`, `{theme}`, "
                "`{off_topic}`, `{output_language}`, `{n_themes}`. Keep the line "
                "that sets the reply format, or the answers cannot be read. Clear a "
                "box to restore the original."
            )
        )
        prompts = dict(old["prompts"])
        titles = {
            "code_single": tr("One theme per passage"),
            "code_multi": tr("Yes or no for every theme"),
            "themes": tr("Propose themes"),
            "nonverbal": tr("Non-verbal cues"),
            "translate": tr("Translate"),
        }
        for step, title in titles.items():
            text = st.text_area(title, prompts.get(step, pl.PROMPTS[step]), height=220)
            if text.strip() and text != pl.PROMPTS[step]:
                prompts[step] = text
            else:
                prompts.pop(step, None)
        new["prompts"] = prompts
    save_if_changed(folder, old, new)
    if themes:
        next_step("transcripts", tr("add and prepare the transcripts"))


def transcripts_page() -> None:
    """Add transcripts, de-identify them and review the result."""
    page_header(
        tr("Transcripts"),
        tr(
            "Add the transcripts, prepare them on this computer, then read the "
            "result before anything is coded."
        ),
    )
    folder, old = current()
    new = dict(old)
    store = folder / "transcripts"
    store.mkdir(exist_ok=True)
    kinds = [t.lstrip(".") for t in pl.TRANSCRIPT_TYPES]

    with st.expander(tr("Add transcripts"), expanded=not pl.list_transcripts(folder)):
        with st.form("add_files", clear_on_submit=True, border=False):
            uploads = st.file_uploader(
                tr("Transcript files"),
                type=kinds,
                accept_multiple_files=True,
                help=tr(
                    "Markdown or plain text with one `SPEAKER: text` turn per "
                    "line works best. Word and PDF files are converted on this "
                    "computer."
                ),
            )
            replace = st.checkbox(tr("Replace files that have the same name"))
            if st.form_submit_button(tr("Add to project"), type="primary"):
                added, skipped = 0, []
                for upload in uploads:
                    target = store / Path(upload.name).name
                    if target.exists() and not replace:
                        skipped.append(target.name)
                        continue
                    target.write_bytes(upload.getvalue())
                    added += 1
                if added:
                    notice(
                        "info",
                        tr("**Added {x0}.**").format(x0=count(added, "transcript")),
                    )
                if skipped:
                    notice(
                        "warning",
                        tr(
                            "**Not added, already in the project:** {x0}. Tick *Replace files "
                            "that have the same name* to overwrite them."
                        ).format(x0=", ".join(skipped)),
                    )
                if not uploads:
                    notice("warning", tr("Choose at least one file first."))
                st.rerun()
        with st.form("add_folder", border=False):
            source = st.text_input(
                tr("Or the path of a folder that holds the transcripts"),
                placeholder=tr(r"C:\study\transcripts"),
            )
            if st.form_submit_button(tr("Add every transcript in the folder")):
                path = Path(source.strip().strip('"'))
                found = (
                    [
                        p
                        for p in path.glob("*")
                        if p.suffix.lower() in pl.TRANSCRIPT_TYPES
                    ]
                    if source.strip() and path.is_dir()
                    else []
                )
                fresh = [p for p in found if not (store / p.name).exists()]
                if not found:
                    st.error(
                        tr(
                            "**No transcripts were found there.** Check the folder path. The "
                            "app reads {x0} files."
                        ).format(x0=", ".join(pl.TRANSCRIPT_TYPES))
                    )
                else:
                    for item in fresh:
                        shutil.copy(item, store)
                    notice(
                        "info",
                        tr("**Added {x0}.**").format(
                            x0=count(len(fresh), "transcript")
                        ),
                    )
                    if len(fresh) < len(found):
                        notice(
                            "warning",
                            tr(
                                "**{x0} skipped:** a transcript with the same name is already in "
                                "the project. Remove it first to replace it."
                            ).format(x0=count(len(found) - len(fresh), "file")),
                        )
                    st.rerun()

    transcripts = pl.list_transcripts(folder)
    if not transcripts:
        st.info(
            tr(
                "**No transcripts yet.** Add files above. The project keeps its own "
                "copy; your files are not changed."
            ),
            icon=":material/upload_file:",
        )
        st.stop()

    def state(transcript: Path) -> str:
        out = pl.result_dir(folder, transcript)
        if not (out / "passages.csv").exists():
            return tr("Not prepared")
        if pl.needs_prepare(folder, old, transcript):
            return tr("Changed since prepared")
        parts = [
            tr("Prepared"),
            tr("reviewed")
            if pl.is_reviewed(folder, transcript)
            else tr("not reviewed"),
        ]
        if (out / "coded.csv").exists():
            parts.append(tr("coded"))
        return ", ".join(parts)

    states = {x.name: state(x) for x in transcripts}
    st.dataframe(
        pd.DataFrame({tr("Transcript"): states.keys(), tr("Status"): states.values()}),
        hide_index=True,
        width="stretch",
    )
    with st.expander(tr("Remove transcripts from the project")):
        doomed = st.multiselect(tr("Transcripts to remove"), list(states))
        if st.button(tr("Remove"), disabled=not doomed, icon=":material/delete:"):

            def remove() -> None:
                for name in doomed:
                    (store / name).unlink(missing_ok=True)
                    shutil.rmtree(folder / "results" / name, ignore_errors=True)
                notice(
                    "info",
                    tr("**Removed {x0}.**").format(x0=count(len(doomed), "transcript")),
                )

            confirm_dialog(
                tr("Remove transcripts"),
                tr(
                    "This removes {x0} from the project, with their passages and "
                    "codes. Your own files are not touched."
                ).format(x0=count(len(doomed), "transcript")),
                tr("Remove"),
                remove,
            )

    st.header(tr("Prepare"), anchor=False)
    st.markdown(
        tr(
            "Preparing splits each transcript into passages and de-identifies them: "
            "names of people and places, phone numbers and email addresses are "
            "replaced with placeholders such as `{tag}`, and speakers are "
            "numbered. **This runs on this computer.** The first time, it downloads "
            "a detection model."
        ).format(tag=TAGS[old["placeholders"]])
    )
    with st.container(border=True):
        new["detector"] = st.selectbox(
            tr("Detector"),
            list(DETECTORS),
            index=list(DETECTORS).index(old["detector"]),
            format_func=DETECTORS.get,
        )
        if new["detector"] != "none":
            new["placeholders"] = st.selectbox(
                tr("Language of the placeholders"),
                list(TAGS),
                index=list(TAGS).index(old["placeholders"]),
                format_func=lambda code: f"{languages().get(code, code)}: {TAGS[code]}",
                help=tr(
                    "The tags that replace names and numbers. Choose the "
                    "language of the transcript so the text reads naturally."
                ),
            )
        saved_roster = "\n".join(read_roster(folder))
        roster = st.text_area(
            tr("Participant names, one per line (optional, stays on this computer)"),
            saved_roster,
            help=tr(
                "A roster catches names the detector misses. It is saved in "
                "the project folder and never sent to a model."
            ),
            disabled=new["detector"] == "none",
        )
        if new["detector"] != "none":
            keep = st.text_area(
                tr("Places and words to keep, one per line (optional)"),
                "\n".join(old["keep_terms"]),
                help=tr(
                    "The detector replaces every place name and cannot tell a "
                    "village from a capital. List the places too large to identify "
                    "anyone (the country, the region, a big city) to leave them in. "
                    "Leave out villages, neighborhoods and anything else small."
                ),
            )
            new["keep_terms"] = [k.strip() for k in keep.splitlines() if k.strip()]
        if new["detector"] == "gliner":
            new["proper_nouns_only"] = st.checkbox(
                tr("Only replace names written with a capital letter"),
                bool(old["proper_nouns_only"]),
                help=tr(
                    "The detector also tags ordinary words such as 'esposo', "
                    "'neighbor', 'casa' or 'market', which identify no one. They "
                    "are written in lowercase, so this leaves them in. Untick it "
                    "for transcripts typed without capital letters, where a name "
                    "looks like any other word. Names in the roster are always "
                    "replaced."
                ),
            )
            new["detector_threshold"] = st.slider(
                tr("Detector sensitivity"),
                0.1,
                0.9,
                float(old["detector_threshold"]),
                0.05,
                help=tr("Lower replaces more, including some ordinary words."),
            )
    if "\n".join(line.strip() for line in roster.splitlines() if line.strip()) != (
        saved_roster
    ):
        (folder / "roster.txt").write_text(roster, "utf-8")
        pl.save_settings(folder, new)
        notice("toast", tr("Saved"))
        st.rerun()
    save_if_changed(folder, old, new)

    todo = [t for t in transcripts if pl.needs_prepare(folder, old, t)]
    coded = [
        t for t in transcripts if (pl.result_dir(folder, t) / "coded.csv").exists()
    ]
    recoded = [t for t in todo if t in coded]

    # The work is done here, on the page, not inside the confirmation dialog:
    # a dialog's Cancel or close button would stop it halfway.
    if "_prepare" in st.session_state:
        force = st.session_state.pop("_prepare")
        clean = cached_cleaner(
            old["detector"],
            tuple(read_roster(folder)),
            old["detector_threshold"],
            tuple(old["keep_terms"]),
            old["proper_nouns_only"],
            old["placeholders"],
        )
        bar = st.progress(0.0, text=tr("Starting..."))
        errors = pl.prepare(
            folder,
            old,
            clean,
            lambda f, t: bar.progress(f, text=t),
            force=force,
            explain=explain,
            say=tr,
        )
        done = (len(transcripts) if force else len(todo)) - len(errors)
        st.session_state["_prepare_errors"] = errors
        if done:
            notice(
                "info",
                tr(
                    "**Prepared {x0}.** Read them below, then mark each one as "
                    "reviewed."
                ).format(x0=count(done, "transcript")),
            )
        st.rerun()
    show_errors(
        st.session_state.pop("_prepare_errors", {}), tr("could not be prepared")
    )

    kept = tr(
        " Model answers already received are kept, and reused wherever the "
        "passage text is unchanged."
    )

    def start(force: bool) -> None:
        st.session_state["_prepare"] = force

    if todo:
        if st.button(
            tr("Prepare {x0}").format(x0=count(len(todo), "new or changed transcript")),
            type="primary",
            icon=":material/shield:",
        ):
            if recoded:
                confirm_dialog(
                    tr("Prepare changed transcripts"),
                    tr(
                        "{x0} of these already have codes. Preparing again replaces the "
                        "passages and deletes those codes, the proposed themes and the "
                        "reviewed mark.{kept}"
                    ).format(x0=count(len(recoded), "transcript"), kept=kept),
                    tr("Prepare"),
                    lambda: start(False),
                )
            else:
                start(False)
                st.rerun()
    else:
        st.markdown(
            tr("All {x0} are prepared.").format(
                x0=count(len(transcripts), "transcript")
            )
        )
        if st.button(tr("Prepare all again"), icon=":material/restart_alt:"):
            confirm_dialog(
                tr("Prepare all transcripts again"),
                tr(
                    "This replaces the passages of all {x0} and deletes the codes of "
                    "{x1}, with their proposed themes and reviewed marks.{kept}"
                ).format(
                    x0=count(len(transcripts), "transcript"), x1=len(coded), kept=kept
                ),
                tr("Prepare all"),
                lambda: start(True),
            )

    done = [
        t for t in transcripts if (pl.result_dir(folder, t) / "passages.csv").exists()
    ]
    if not done:
        st.stop()
    st.header(tr("Review before coding"), anchor=False)
    st.warning(
        tr(
            "**Detectors miss things.** Read every passage and look for names, "
            "places and numbers that are still there. The list of replacements "
            "shows only what was found. To fix a miss, add the name to the roster "
            "above and prepare again."
        ),
        icon=":material/visibility:",
    )
    choice = st.selectbox(
        tr("Transcript"),
        done,
        format_func=lambda p: f"{p.name} · "
        + (tr("reviewed") if pl.is_reviewed(folder, p) else tr("not reviewed")),
    )
    out = pl.result_dir(folder, choice)
    passages = pl.read_table(out / "passages.csv")
    original = original_passages(
        str(choice),
        choice.stat().st_mtime,
        old["unit"],
        tuple(old["moderator_labels"]),
        old["participant_label"],
    )
    if len(original) != len(passages) or pl.needs_prepare(folder, old, choice):
        st.info(
            tr(
                "**This transcript or its settings changed since it was prepared.** "
                "Prepare it again to compare it with the original."
            ),
            icon=":material/sync_problem:",
        )
        st.dataframe(passages[["passage_id", "speaker", "text"]], hide_index=True)
    else:
        review = passages[["passage_id", "speaker"]].assign(
            original=original["text"].to_numpy(), cleaned=passages["text"]
        )
        changed = review["original"] != review["cleaned"]
        only_changed = st.toggle(
            tr("Show only passages where something was replaced ({x0})").format(
                x0=int(changed.sum())
            )
        )
        shown = review[changed] if only_changed else review
        key = f"review_{choice.name}_{only_changed}"
        for i in page_rows(len(shown), key):
            row = shown.iloc[i]
            with st.container(border=True):
                if row.original == row.cleaned:
                    st.caption(
                        tr("Passage {x0} · {x1} · nothing replaced").format(
                            x0=row.passage_id, x1=row.speaker
                        )
                    )
                    st.markdown(md_escape(row.cleaned))
                else:
                    st.caption(
                        tr("Passage {x0} · {x1}").format(
                            x0=row.passage_id, x1=row.speaker
                        )
                    )
                    left, right = st.columns(2)
                    left.markdown(
                        tr("**Original**  \n{x0}").format(x0=md_escape(row.original))
                    )
                    right.markdown(
                        tr("**De-identified**  \n{x0}").format(
                            x0=highlight_placeholders(row.cleaned)
                        )
                    )
        pager(len(shown), key)
        st.caption(
            tr("{x0} of {x1} shown").format(
                x0=count(len(shown), "passage"), x1=len(review)
            )
        )
        if pl.is_reviewed(folder, choice):
            st.info(
                tr("**{x0} is marked as reviewed.**").format(x0=choice.name),
                icon=":material/task_alt:",
            )
            if st.button(tr("Undo the reviewed mark")):
                pl.set_reviewed(folder, choice, False)
                st.rerun()
        elif st.button(
            tr("I have read every passage: mark {x0} as reviewed").format(
                x0=choice.name
            ),
            type="primary",
            icon=":material/task_alt:",
        ):
            pl.set_reviewed(folder, choice, True)
            st.rerun()
    findings = pd.read_csv(out / "findings.csv")
    with st.expander(
        tr("What was replaced ({x0})").format(x0=count(len(findings), "item"))
    ):
        st.markdown(tr("This list holds the original names. It stays in the project."))
        st.dataframe(findings, hide_index=True, width="stretch")
    if not todo:
        next_step("run", tr("choose the steps and run them"))


def run_page() -> None:
    """Choose the steps and run them on every prepared transcript."""
    page_header(
        tr("Run"),
        tr("Choose what to do with the prepared passages, then run it on all of them."),
    )
    folder, old = current()
    new = dict(old)
    status = pl.project_status(folder, old)
    ready = status["prepared"] + status["stale"]
    if not ready:
        st.info(
            tr(
                "**No transcript is prepared yet.** Add and prepare transcripts on "
                "the **Transcripts** page first."
            ),
            icon=":material/shield:",
        )
        st.stop()
    metric_row(
        [
            (tr("Prepared transcripts"), ready),
            (tr("Passages"), f"{status['passages']:,}"),
            (tr("Themes in the codebook"), status["themes"]),
        ]
    )
    if status["stale"]:
        st.warning(
            tr(
                "**{x0} changed since being prepared.** The run uses the old "
                "passages. Prepare again on the **Transcripts** page to pick up "
                "the change."
            ).format(x0=count(status["stale"], "transcript")),
            icon=":material/sync_problem:",
        )

    has_model = bool(old["model"])
    saved = [s for s in old["steps"] if s == "embeddings" or has_model]
    chosen: dict[str, bool] = {}
    st.header(tr("Without a language model"), anchor=False)
    chosen["embeddings"] = st.checkbox(
        STEP_LABELS["embeddings"],
        value="embeddings" in saved,
        help=tr(
            "Uses the embedding model, on this computer by default, and costs "
            "nothing. Less accurate than a language model. The first run downloads "
            "about 210 MB."
        ),
    )
    st.header(tr("With the language model"), anchor=False)
    if has_model:
        st.markdown(
            tr("{x0}, model **{x1}**.").format(
                x0=BACKENDS[old["backend"]], x1=old["model"]
            )
        )
    else:
        st.info(
            tr(
                "**No language model is chosen**, so these steps are switched off. "
                "Choose one on the **Project and models** page to use them."
            ),
            icon=":material/smart_toy:",
        )
    for step in ("llm", "llm2", "nonverbal", "themes", "translate"):
        label = STEP_LABELS[step]
        if step == "translate":
            label = tr("Translate the passages into {x0} first").format(
                x0=old["output_language"]
            )
        chosen[step] = st.checkbox(label, value=step in saved, disabled=not has_model)
    steps = [step for step in pl.STEPS if chosen.get(step)]
    if "llm2" in steps:
        new["model2"] = st.text_input(
            tr("Second model"),
            old["model2"],
            help=tr("Another model on the same service ({x0}).").format(
                x0=BACKENDS[old["backend"]]
            ),
        )
    # Remember the ticks with the project. Without a model, the switched-off
    # steps keep whatever was saved, ready for when one is chosen.
    new["steps"] = [
        s
        for s in pl.STEPS
        if (chosen[s] if has_model or s == "embeddings" else s in old["steps"])
    ]
    if new != old:
        pl.save_settings(folder, new)

    problems = []
    needs_model = any(step != "embeddings" for step in steps)
    if not steps:
        problems.append(tr("Tick at least one step."))
    if {"embeddings", "llm", "llm2"} & set(steps) and not old["themes"]:
        problems.append(
            tr("**The codebook is empty.** Add themes on the **Study inputs** page.")
        )
    if "llm2" in steps and not new["model2"]:
        problems.append(tr("Type the name of the second model."))
    for problem in problems:
        st.warning(problem)

    st.header(tr("Where the text goes"), anchor=False)
    confirmed = True
    destinations = cloud_destinations(new, steps)
    if destinations:
        names = " and ".join(name for _, name in destinations)
        level = approval(destinations)
        prepared = (
            tr("the passages as you added them (this app did not de-identify them)")
            if old["detector"] == "none"
            else tr("the de-identified passages")
        )
        notes = " ".join(
            f"{name[0].upper()}{name[1:]}: {approval_note(key).lower()}"
            for key, name in destinations
        )
        message = tr(
            "**These steps send text to {names}.** That is {prepared}, the "
            "moderator questions, your research question and study context, "
            "and the codebook with any example quotes.\n\n{notes}"
        ).format(names=names, prepared=prepared, notes=notes)
        (st.warning if level else st.error)(message, icon=":material/cloud_upload:")
        unreviewed = ready - status["reviewed"]
        if unreviewed:
            st.warning(
                tr(
                    "**{x0} of {ready} not marked as reviewed.** Read the passages on "
                    "the **Transcripts** page and mark each transcript before sending "
                    "it out."
                ).format(x0=count(unreviewed, "transcript"), ready=ready),
                icon=":material/visibility:",
            )
        # Keyed by project, destination and the prepared passages: a new
        # project, another service or re-prepared text asks again.
        stamps = sorted(
            (pl.result_dir(folder, t) / "prepared.txt").read_text("utf-8")
            for t in pl.list_transcripts(folder)
            if (pl.result_dir(folder, t) / "prepared.txt").exists()
        )
        gate = hashlib.sha256(
            json.dumps([folder.name, [k for k, _ in destinations], stamps]).encode()
        ).hexdigest()[:12]
        classified = st.checkbox(
            tr("This data is classified {x0}.").format(x0=LEVELS[level])
            if level
            else tr("I am allowed to send this data to {names}.").format(names=names),
            key=f"class_{gate}",
        )
        read = st.checkbox(
            tr("I have read the prepared passages of all {x0}.").format(
                x0=count(ready, "transcript")
            ),
            key=f"read_{gate}",
        )
        confirmed = classified and read
    else:
        st.success(
            tr("**These steps run on this computer.** No text leaves it."),
            icon=":material/computer:",
        )
    requests = pl.estimate_requests(status, new, steps)
    if requests:
        st.markdown(
            tr("This sends up to **{x0}** to the model.").format(
                x0=count(requests, "request")
            )
        )
    st.markdown(
        tr(
            "A run cannot be paused from this page, but stopping is safe: close the "
            "tab or the app, and the answers already received are kept. Run again "
            "and only the missing ones are requested. Do not click elsewhere on the "
            "page while it runs."
        )
    )

    if st.button(
        tr("Run on {x0}").format(x0=count(ready, "transcript")),
        type="primary",
        icon=":material/play_arrow:",
        disabled=bool(problems) or not confirmed,
    ):
        complete = complete2 = embed = None
        try:
            cache = folder / "llm_cache.jsonl"
            if needs_model:
                client, wire = client_for(new, new["backend"])
                complete = pl.Completer(client, wire, new["model"], cache)
                complete2 = pl.Completer(client, wire, new["model2"], cache)
            if "embeddings" in steps:
                where = new["embed_backend"]
                embed = pl.make_embedder(
                    where,
                    new["embed_model"]
                    or ("" if where == "local" else "text-embedding-3-small"),
                    None if where == "local" else client_for(new, where)[0],
                )
        except Exception as exc:  # noqa: BLE001 - show any setup failure
            st.error(explain(exc), icon=":material/error:")
            st.stop()
        record = {
            "state": "running",
            "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "steps": steps,
            "model": new["model"] if needs_model else "",
            "fingerprint": pl.coding_fingerprint(new, steps),
        }
        (folder / "last_run.json").write_text(json.dumps(record), "utf-8")
        bar = st.progress(0.0, text=tr("Starting..."))
        errors = pl.run_batch(
            folder,
            new,
            steps,
            complete,
            embed,
            complete2,
            lambda f, t: bar.progress(min(f, 1.0), text=t),
            explain=explain,
            say=tr,
        )
        used = [c for c in (complete, complete2) if c]
        record |= {
            "state": "done",
            "errors": errors,
            "done": ready - len(errors),
            "calls": sum(c.calls for c in used),
            "reused": sum(c.reused for c in used),
            "input": sum(c.input_tokens for c in used),
            "output": sum(c.output_tokens for c in used),
        }
        (folder / "last_run.json").write_text(json.dumps(record), "utf-8")
        if record["done"]:
            notice(
                "info",
                tr(
                    "**Finished: {x0} coded.** Read the passages on the **Results** "
                    "page."
                ).format(x0=count(record["done"], "transcript")),
            )
        st.rerun()

    record = last_run(folder)
    if not record:
        return
    st.header(tr("Last run"), anchor=False)
    did = ", ".join(STEP_NAMES[s] for s in record.get("steps", []))
    if record.get("state") == "running":
        st.warning(
            tr(
                "**The run started on {x0} did not finish** ({did}). It was "
                "interrupted by closing the page or clicking elsewhere. Run again "
                "to complete it: the answers already received are kept."
            ).format(x0=record["when"], did=did),
            icon=":material/pause_circle:",
        )
        return
    model = tr(", with {x0}").format(x0=record["model"]) if record.get("model") else ""
    st.markdown(f"{record['when']}: {did}{model}.")
    show_errors(record["errors"], tr("not finished"))
    if record["errors"]:
        st.markdown(
            tr(
                "Run again to retry. Answers already received are reused, not paid "
                "for twice."
            )
        )
    if codes_outdated(folder, old):
        outdated_warning()
    metric_row(
        [
            (tr("Transcripts coded"), record["done"]),
            (
                tr("Requests"),
                f"{record['calls']:,}",
                tr("Sent to the model in this run."),
            ),
            (
                tr("Reused"),
                f"{record['reused']:,}",
                tr("Answers saved by an earlier run."),
            ),
            (
                tr("Tokens in"),
                f"{record['input']:,}",
                tr(
                    "Pieces of text sent to the model. Services bill by the token; "
                    "a token is about three quarters of a word."
                ),
            ),
            (
                tr("Tokens out"),
                f"{record['output']:,}",
                tr("Pieces of text the model wrote back, billed at a higher rate."),
            ),
        ]
    )
    if record["done"]:
        next_step("results", tr("read the coded passages"))


def code_sources(coded: pd.DataFrame) -> dict[str, tuple[str, dict | None]]:
    """List the sets of codes: label, and theme columns for yes/no coding."""
    sources: dict[str, tuple[str, dict | None]] = {
        column: (label, None)
        for column, label in CODE_COLUMNS.items()
        if column in coded.columns
    }
    for prefix, label in (("llm", tr("Language model")), ("llm2", tr("Second model"))):
        themes = {
            c[len(prefix) + 2 :]: c
            for c in coded.columns
            if c.startswith(f"{prefix}: ")
        }
        if themes:
            sources[f"{prefix}_multi"] = (
                tr("{label}, yes or no per theme").format(label=label),
                themes,
            )
    return sources


def assigned_themes(coded: pd.DataFrame, source: str, sources: dict) -> pd.Series:
    """Themes each passage has under one set of codes, as a list per passage."""
    themes = sources[source][1]
    if themes is None:
        return coded[source].map(lambda value: [str(value)])
    rows = [
        [name for name, column in themes.items() if row[column] == "YES"]
        for _, row in coded.iterrows()
    ]
    return pd.Series(rows, index=coded.index, dtype=object)


def results_page() -> None:
    """Browse and download the coded passages and proposed themes."""
    page_header(
        tr("Results"),
        tr("Read the coded passages, see how often each theme appears, and download."),
    )
    folder, old = current()
    coded = pl.load_coded(folder)
    if coded.empty:
        st.info(
            tr(
                "**Nothing is coded yet.** Choose the steps and run them on the "
                "**Run** page."
            ),
            icon=":material/play_arrow:",
        )
        st.stop()
    if codes_outdated(folder, old):
        outdated_warning()

    names = sorted(coded["transcript"].unique())
    picked = st.multiselect(
        tr("Transcripts"),
        names,
        placeholder=tr("Showing all. Choose some to narrow down."),
    )
    view = coded[coded["transcript"].isin(picked)] if picked else coded
    sources = code_sources(view)
    no_theme = tr("No theme")
    metric_row(
        [
            (tr("Transcripts"), view["transcript"].nunique()),
            (tr("Passages"), f"{len(view):,}"),
            (
                tr("Coding steps run"),
                len(sources),
                "; ".join(label for label, _ in sources.values()) or tr("None yet."),
            ),
        ]
    )

    if sources:
        st.header(tr("Passages per theme"), anchor=False)
        source = st.radio(
            tr("Codes from"),
            list(sources),
            format_func=lambda key: sources[key][0],
            horizontal=True,
        )
        lists = {key: assigned_themes(view, key, sources) for key in sources}
        long = pd.DataFrame(
            {
                "Theme": lists[source].map(lambda found: found or [no_theme]).explode(),
                "transcript": view["transcript"],
            }
        )
        counts = pd.crosstab(long["Theme"], long["transcript"], margins=True)
        counts = counts.rename(index={"All": tr("Total")}, columns={"All": tr("Total")})
        counts = counts.rename_axis(index=tr("Theme"), columns=None)
        st.dataframe(counts, width="stretch")
        if sources[source][1]:
            st.caption(
                tr(
                    "A passage can count under several themes, so the totals can "
                    "exceed the number of passages."
                )
            )

        st.header(tr("Read the passages"), anchor=False)
        theme = st.pills(
            tr("Theme ({x0})").format(x0=sources[source][0].lower()),
            sorted(set(long["Theme"])),
            key=f"pill_{source}",
        )
        search = st.columns([1, 2])[0].text_input(
            tr("Find words in the passages"), key="search"
        )
        singles = [key for key, (_, themes) in sources.items() if themes is None]
        differ = pd.Series(False, index=view.index)
        only_differ = False
        if len(singles) > 1:
            differ = view[singles].nunique(axis=1) > 1
            only_differ = st.toggle(
                tr("Show only passages the coding steps disagree on ({x0})").format(
                    x0=int(differ.sum())
                )
            )
        keep = pd.Series(True, index=view.index)
        if theme:
            keep &= lists[source].map(lambda found: theme in (found or [no_theme]))
        if search.strip():
            keep &= view["text"].str.contains(search.strip(), case=False, regex=False)
        if only_differ:
            keep &= differ
        quotes = view[keep]
        key = f"quotes_{source}_{theme}_{search}_{only_differ}_{len(quotes)}"
        if quotes.empty:
            st.info(
                tr("**No passage matches.** Clear the theme, the words or the toggle.")
            )
        for i in page_rows(len(quotes), key):
            row = quotes.iloc[i]
            with st.container(border=True):
                speaker = f" · {row.speaker}" if row.speaker else ""
                flag = tr(" · coding steps disagree") if differ.loc[row.name] else ""
                st.caption(
                    tr("{x0} · passage {x1}{speaker}{flag}").format(
                        x0=row.transcript, x1=row.passage_id, speaker=speaker, flag=flag
                    )
                )
                st.markdown(highlight_placeholders(str(row.text)))
                for name, (label, themes) in sources.items():
                    found = lists[name].loc[row.name] or [no_theme]
                    st.markdown(f"**{md_escape(', '.join(found))}**")
                    reason = ""
                    if themes is None:
                        reason = str(row.get(name.replace("_theme", "_rationale"), ""))
                    st.caption(f"{label}{': ' + reason if reason.strip() else ''}")
        pager(len(quotes), key)
        st.caption(
            tr("{x0} of {x1:,} shown").format(
                x0=count(len(quotes), "passage"), x1=len(view)
            )
        )

    if {"llm_theme", "llm2_theme"} <= set(coded.columns):
        st.header(tr("Do the two models agree?"), anchor=False)
        result = pl.agreement(
            view["llm_theme"], view["llm2_theme"], old["model"], old["model2"]
        )
        metric_row(
            [
                (tr("Passages coded the same"), f"{result['agreement']:.0%}"),
                (
                    tr("Cohen's kappa"),
                    f"{result['kappa']:.2f}",
                    tr(
                        "Agreement beyond what chance would give: 0 is chance, 1 is "
                        "identical. Above about 0.6 is usually read as substantial."
                    ),
                ),
            ]
        )
        st.dataframe(result["confusion"], width="stretch")

    proposed: dict[str, dict] = {}
    for transcript in pl.list_transcripts(folder):
        path = pl.result_dir(folder, transcript) / "themes.json"
        if path.exists():
            for item in json.loads(path.read_text("utf-8")):
                if item["name"] == pl.UNPARSED:
                    continue
                entry = proposed.setdefault(
                    item["name"].casefold(), {**item, "from": []}
                )
                entry["from"].append(transcript.name)
    if proposed:
        st.header(tr("Proposed themes"), anchor=False)
        st.markdown(
            tr(
                "Themes the language model proposed. Tick rows to add them to the "
                "codebook, then run again to code with them."
            )
        )
        table = pd.DataFrame(
            [{**item, "from": ", ".join(item["from"])} for item in proposed.values()]
        )
        event = st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="multi-row",
            column_config={
                "name": tr("Name"),
                "definition": tr("Definition"),
                "examples": tr("Example quotes"),
                "from": tr("Proposed from"),
            },
        )
        rows = event.selection.rows
        if st.button(tr("Add selected to the codebook"), disabled=not rows):
            have = {t["name"].casefold() for t in old["themes"]}
            chosen = table.iloc[rows].drop(columns="from").to_dict("records")
            fresh = [t for t in chosen if t["name"].casefold() not in have]
            themes = themes_from_data([*old["themes"], *fresh])
            pl.save_settings(folder, {**old, "themes": themes_to_data(themes)})
            already = len(chosen) - len(fresh)
            notice(
                "info",
                tr("**Added {x0} to the codebook.** ").format(
                    x0=count(len(fresh), "theme")
                )
                + (
                    tr("{already} already in it. ").format(already=already)
                    if already
                    else ""
                )
                + tr("Run again on the **Run** page to code with the new themes."),
            )
            st.rerun()

    with st.expander(tr("All columns as a table")):
        st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            column_config={c: COLUMN_LABELS.get(c, c) for c in view.columns},
        )

    st.header(tr("Download"), anchor=False)
    left, right = st.columns(2)
    left.download_button(
        tr("Coded passages (CSV)"),
        view.to_csv(index=False).encode("utf-8-sig"),
        "coded_passages.csv",
        "text/csv",
        icon=":material/download:",
    )
    right.download_button(
        tr("Whole project (zip)"),
        lambda: pl.export_zip(folder),
        f"{folder.name}.zip",
        "application/zip",
        icon=":material/download:",
        help=tr(
            "Settings, codebook and results. Leaves out the original "
            "transcripts, the roster and the lists of replaced names."
        ),
    )
    next_step("human", tr("check the codes against a person's codes"))


def human_page() -> None:
    """Compare model codes with a person's codes for the same passages."""
    page_header(
        tr("Check against human codes"),
        tr(
            "Have a person code a sample of passages, then see how often the model "
            "chose the same theme. Do this before trusting the model's codes."
        ),
    )
    folder, old = current()
    coded = pl.load_coded(folder)
    present = [c for c in CODE_COLUMNS if c in coded.columns]
    if not present:
        st.info(
            tr(
                "**No single-theme codes yet.** This check needs one theme per "
                "passage. Run the similarity pass, or the language model in *one "
                "theme per passage* mode, on the **Run** page."
            ),
            icon=":material/play_arrow:",
        )
        st.stop()
    allowed = [t["name"] for t in old["themes"]] + [old["off_topic"]]
    st.markdown(
        tr(
            "1. Download the template: every passage, with an empty `theme` column.\n"
            "2. Fill in `theme` for the passages the person coded. Leave the others "
            "empty.\n"
            "3. Upload the file."
        )
    )
    st.markdown(tr("Theme names to use: ") + ", ".join(f"`{name}`" for name in allowed))
    template = coded[["transcript", "passage_id", "speaker", "text"]].assign(theme="")
    st.download_button(
        tr("Download the coding template (CSV)"),
        template.to_csv(index=False).encode("utf-8-sig"),
        "human_codes_template.csv",
        "text/csv",
        icon=":material/download:",
    )
    example = next((e for e in EXAMPLES.values() if e["project"] == folder.name), None)
    if example:
        file = example["human"]
        st.caption(
            tr(
                "For this example, a person's codes are ready in "
                "`data/fictitious/{file}`."
            ).format(file=file)
        )
    upload = st.file_uploader(tr("Human codes (CSV)"), type=["csv"])
    if not upload:
        st.stop()
    try:
        human = pd.read_csv(upload, keep_default_na=False, encoding="utf-8-sig")
        human["passage_id"] = pd.to_numeric(human["passage_id"]).astype(int)
    except KeyError:
        st.error(tr("**Columns missing.** The file needs `passage_id` and `theme`."))
        st.stop()
    except Exception:  # noqa: BLE001 - any unreadable file
        st.error(
            tr(
                "**This file could not be read.** Save it as CSV (UTF-8), with whole "
                "numbers in `passage_id`, and try again."
            )
        )
        st.stop()
    if "theme" not in human.columns:
        st.error(tr("**Columns missing.** The file needs `passage_id` and `theme`."))
        st.stop()
    human = human[human["theme"].astype(str).str.strip() != ""]
    human = human[[c for c in ("transcript", "passage_id", "theme") if c in human]]
    if "transcript" not in human.columns:
        if coded["transcript"].nunique() > 1:
            st.error(
                tr(
                    "**Add a `transcript` column** with the file name of each "
                    "passage's transcript: this project has several."
                )
            )
            st.stop()
        human["transcript"] = coded["transcript"].iloc[0]
    merged = human.merge(coded, on=["transcript", "passage_id"], how="inner")
    if merged.empty:
        st.error(
            tr(
                "**No row matches a coded passage.** Check the `transcript` and "
                "`passage_id` values against the template, and that `theme` is "
                "filled in."
            )
        )
        st.stop()
    if len(merged) < len(human):
        st.warning(
            tr(
                "**{x0} left out:** no coded passage has that transcript and "
                "passage number."
            ).format(x0=count(len(human) - len(merged), "row"))
        )
    person = merged["theme"].astype(str).str.strip()
    unknown = sorted(set(person) - set(allowed))
    if unknown:
        st.warning(
            tr(
                "**Not in the codebook:** {x0}. These can never match a model "
                "code. Check the spelling."
            ).format(x0=", ".join(unknown))
        )
    if codes_outdated(folder, old):
        outdated_warning()
    column = st.radio(
        tr("Compare with"), present, format_func=CODE_COLUMNS.get, horizontal=True
    )
    result = pl.agreement(person, merged[column], tr("Human"), CODE_COLUMNS[column])
    metric_row(
        [
            (tr("Passages compared"), result["n"]),
            (tr("Coded the same"), f"{result['agreement']:.0%}"),
            (
                tr("Cohen's kappa"),
                f"{result['kappa']:.2f}",
                tr(
                    "Agreement beyond what chance would give: 0 is chance, 1 is "
                    "identical. Above about 0.6 is usually read as substantial."
                ),
            ),
        ]
    )
    st.markdown(tr("Rows are the person's codes, columns the model's."))
    st.dataframe(result["confusion"], width="stretch")
    st.header(tr("Passages coded differently"), anchor=False)
    differ = merged.loc[result["disagreements"]]
    key = f"differ_{column}_{len(differ)}"
    for i in page_rows(len(differ), key):
        row = differ.iloc[i]
        with st.container(border=True):
            st.caption(
                tr("{x0} · passage {x1}").format(x0=row.transcript, x1=row.passage_id)
            )
            st.markdown(highlight_placeholders(str(row.text)))
            st.markdown(f"**{md_escape(str(row.theme))}**")
            st.caption(tr("Human"))
            st.markdown(f"**{md_escape(str(row[column]))}**")
            st.caption(CODE_COLUMNS[column])
    pager(len(differ), key)
    if differ.empty:
        st.info(tr("**Every compared passage has the same code.**"))
    if column == "embedding_theme":
        st.header(tr("Agreement by relevance threshold"), anchor=False)
        st.markdown(
            tr(
                "Pick the row with the best agreement, then go a little lower: a "
                "passage dropped by mistake is harder to notice than small talk kept "
                "by mistake. Set the value under *Advanced* on **Study inputs** and "
                "run the similarity pass again."
            )
        )
        table = pl.threshold_table(person, merged, old["off_topic"])
        st.dataframe(
            table.round(2),
            hide_index=True,
            column_config={c: c.capitalize() for c in table.columns},
        )


# --- shell ------------------------------------------------------------------


def image_uri(path: Path) -> str:
    """Inline an image so it can sit inside a link."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    """Page setup, navigation and the sidebar."""
    st.set_page_config(page_title=APP_NAME, page_icon=":material/forum:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if "_open" in st.session_state:
        st.session_state["project"] = st.session_state.pop("_open")
    projects_dir().mkdir(parents=True, exist_ok=True)
    start(projects_dir() / "ui.json")
    existing = sorted(p.name for p in projects_dir().iterdir() if p.is_dir())
    if st.session_state.get("project") not in existing:
        st.session_state.pop("project", None)
    name = st.session_state.get("project") or (existing[0] if existing else None)

    settings = None
    done = {"study": False, "transcripts": False, "run": False}
    if name:
        folder = projects_dir() / name
        settings = pl.load_settings(folder)
        s = pl.project_status(folder, settings)
        record = last_run(folder) or {}
        prepared = s["transcripts"] > 0 and s["prepared"] == s["transcripts"]
        done["study"] = s["themes"] > 0
        done["transcripts"] = prepared
        # A check on Run means the codes on screen match the current inputs.
        done["run"] = (
            prepared
            and s["coded"] == s["transcripts"]
            and record.get("state") == "done"
            and not record.get("errors")
            and not codes_outdated(folder, settings)
        )

    def icon(page: str, otherwise: str) -> str:
        return ":material/check_circle:" if done[page] else otherwise

    PAGES.clear()
    PAGES["project"] = st.Page(
        project_page,
        title=tr("1. Project and models"),
        icon=":material/folder_open:",
        default=True,
    )
    if name:
        PAGES["study"] = st.Page(
            study_page,
            title=tr("2. Study inputs"),
            icon=icon("study", ":material/edit_note:"),
        )
        PAGES["transcripts"] = st.Page(
            transcripts_page,
            title=tr("3. Transcripts"),
            icon=icon("transcripts", ":material/shield:"),
        )
        PAGES["run"] = st.Page(
            run_page, title=tr("4. Run"), icon=icon("run", ":material/play_arrow:")
        )
        PAGES["results"] = st.Page(
            results_page, title=tr("5. Results"), icon=":material/table_chart:"
        )
        PAGES["human"] = st.Page(
            human_page, title=tr("6. Check against human codes"), icon=":material/rule:"
        )
    page = st.navigation(list(PAGES.values()))

    # Drawn before page.run(), so a page that stops early cannot skip it.
    with st.sidebar:
        if existing:
            st.selectbox(tr("Project"), existing, key="project")
            destinations = cloud_destinations(settings)
            names = " and ".join(n for _, n in destinations)
            if not destinations:
                st.success(
                    tr("Everything runs on this computer."), icon=":material/computer:"
                )
            elif approval(destinations):
                st.warning(
                    tr(
                        "Passages go to **{names}** when you run a step that uses it."
                    ).format(names=names),
                    icon=":material/cloud_upload:",
                )
            else:
                st.error(
                    tr(
                        "Passages go to **{names}** when you run a step that uses it. No "
                        "approval is recorded for it."
                    ).format(names=names),
                    icon=":material/cloud_upload:",
                )
            st.divider()
        try:
            app_version = version("llm-quali-coding")
        except PackageNotFoundError:
            app_version = "dev"
        st.markdown(f"**{APP_NAME}**")
        st.caption(
            tr("Version {app_version} | [Documentation]({DOCS_URL})").format(
                app_version=app_version, DOCS_URL=DOCS_URL
            )
        )
        logo = ASSETS / "IPA-primary-color-RGB.png"
        if logo.exists():
            st.markdown(
                '<a href="https://www.poverty-action.org" target="_blank" '
                f'rel="noopener noreferrer"><img src="{image_uri(logo)}" '
                'alt="Innovations for Poverty Action (IPA) logo" '
                'style="width:60%;"></a>',
                unsafe_allow_html=True,
            )
        st.caption(
            tr(
                "Help: [{SUPPORT_EMAIL}](mailto:{SUPPORT_EMAIL})  "
                "\n:material/bug_report: [Report an issue]({ISSUES_URL})"
            ).format(SUPPORT_EMAIL=SUPPORT_EMAIL, ISSUES_URL=ISSUES_URL)
        )
        st.selectbox(
            LANGUAGE_LABEL, list(languages()), format_func=languages().get, key="lang"
        )
        remember(projects_dir() / "ui.json")
    page.run()


if __name__ == "__main__":
    main()
