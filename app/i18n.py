"""Interface language for the web app.

Every string a person reads goes through `tr()`. The English text is the key:
`tr("Run")` returns "Ejecutar" when the interface is in Spanish, and "Run" when
no translation exists, so a missing entry never breaks a page.

To add a language, copy `locales/es.json` to `locales/<code>.json`, set
`"_name"` to the language's own name, and translate the values. It then
appears in the language list. `tests/test_i18n.py` reports any string a
catalog is missing.
"""

from __future__ import annotations

import ast
import json
from functools import cache
from pathlib import Path

import streamlit as st

LOCALES = Path(__file__).resolve().parent / "locales"


@cache
def catalog(code: str) -> dict[str, str]:
    """Load the translations for one language; empty when there is no file."""
    path = LOCALES / f"{code}.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def languages() -> dict[str, str]:
    """Available interface languages: code -> the language's own name."""
    found = {"en": "English"}
    for path in sorted(LOCALES.glob("*.json")):
        found[path.stem] = catalog(path.stem).get("_name", path.stem)
    return found


def language() -> str:
    """Code of the language the interface is shown in."""
    return st.session_state.get("lang", "en")


def start(settings_file: Path) -> None:
    """Choose the language for a new session.

    The last choice made on this computer wins; otherwise the browser's
    language when the app has it; otherwise English.
    """
    if "lang" in st.session_state:
        return
    code = ""
    if settings_file.exists():
        code = json.loads(settings_file.read_text("utf-8")).get("lang", "")
    if not code:
        code = (st.context.locale or "en")[:2].lower()
    st.session_state["lang"] = code if code in languages() else "en"


def remember(settings_file: Path) -> None:
    """Save the current language as this computer's choice."""
    wanted = json.dumps({"lang": language()})
    if not settings_file.exists() or settings_file.read_text("utf-8") != wanted:
        settings_file.write_text(wanted, "utf-8")


def tr(text: str) -> str:
    """Return `text` in the interface language."""
    return catalog(language()).get(text, text)


def N_(text: str) -> str:  # noqa: N802 - the conventional name for this marker
    """Mark a constant for translation without translating it yet.

    Constants are defined before the language is known; they are translated
    where they are shown, by `tr()` or by a `Tr` table.
    """
    return text


class Tr(dict):
    """Lookup table whose values come back in the interface language."""

    def __getitem__(self, key):
        """Return the translated value for a key."""
        return tr(super().__getitem__(key))

    def get(self, key, default=None):
        """Return the translated value for a key, or the default."""
        return tr(super().__getitem__(key)) if key in self else default

    def items(self):
        """Return pairs of key and translated value."""
        return [(key, self[key]) for key in self]


def source_strings(*files: Path) -> set[str]:
    """Every string marked for translation in the given Python files.

    Finds the constant first argument of `tr()`, `N_()` and `say()` calls.
    """
    found: set[str] = set()
    for file in files:
        for node in ast.walk(ast.parse(file.read_text("utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"tr", "N_", "say"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.add(node.args[0].value)
    return found
