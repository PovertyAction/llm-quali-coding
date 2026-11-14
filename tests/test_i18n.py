"""The interface languages are complete, and each language has a working example."""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import i18n  # noqa: E402

from src import pipeline as pl  # noqa: E402
from src.coding import load_themes  # noqa: E402

FIELD = re.compile(r"\{[^{}]*\}")


def source_strings() -> set[str]:
    """Every string the app can show: marked in the code, or a pipeline label."""
    marked = i18n.source_strings(ROOT / "app" / "main.py", ROOT / "src" / "pipeline.py")
    return marked | set(pl.STEPS.values())


@pytest.mark.parametrize("code", [c for c in i18n.languages() if c != "en"])
def test_catalog_is_complete_and_keeps_the_placeholders(code):
    catalog = i18n.catalog(code)
    assert catalog["_name"]
    missing = sorted(source_strings() - set(catalog))
    assert not missing, f"{code}.json has no translation for: {missing[:5]}"
    for english, translated in catalog.items():
        if english == "_name":
            continue
        assert translated.strip(), english
        # A renamed {placeholder} would crash the page that fills it in.
        assert sorted(FIELD.findall(english)) == sorted(FIELD.findall(translated)), (
            english
        )
        assert english.count("`") == translated.count("`"), english
        edges = (
            len(english) - len(english.lstrip()),
            len(english) - len(english.rstrip()),
        )
        assert edges == (
            len(translated) - len(translated.lstrip()),
            len(translated) - len(translated.rstrip()),
        ), english


def test_interface_follows_the_chosen_language(tmp_path, monkeypatch):
    monkeypatch.setenv("QUALI_PROJECTS_DIR", str(tmp_path))
    import main

    main.make_example("fr")
    folder = tmp_path / "example_fr"
    settings = pl.load_settings(folder)
    pl.prepare(folder, settings, pl.make_cleaner("none"))
    embed = lambda texts, query=False: [[1.0, float(len(x) % 2)] for x in texts]  # noqa: E731
    pl.run_batch(folder, settings, ["embeddings"], embed=embed)
    pages = ["project", "study", "transcripts", "run", "results", "human"]
    for code in i18n.languages():
        for page in pages:
            # A translation with a wrong {placeholder} would raise here.
            at = AppTest.from_string(
                f"import main\nmain.{page}_page()", default_timeout=60
            )
            at.session_state["project"] = "example_fr"
            at.session_state["lang"] = code
            at.run()
            assert not at.exception, (code, page, at.exception)
        assert at.title[0].value == i18n.catalog(code).get(
            "Check against human codes", "Check against human codes"
        )
    assert i18n.catalog("es")["Study inputs"] != "Study inputs"


@pytest.mark.parametrize("code", ["en", "es", "fr"])
def test_each_language_has_a_complete_example(code, tmp_path, monkeypatch):
    monkeypatch.setenv("QUALI_PROJECTS_DIR", str(tmp_path))
    import main

    example = main.EXAMPLES[code]
    main.make_example(code)
    folder = tmp_path / example["project"]
    settings = pl.load_settings(folder)
    assert pl.prepare(folder, settings, pl.make_cleaner("none")) == {}
    passages = pl.read_table(
        pl.result_dir(folder, pl.list_transcripts(folder)[0]) / "passages.csv"
    )
    # Five participants, 26 turns, renamed; the moderator's turns are questions.
    assert len(passages) == 26
    assert passages["speaker"].nunique() == 5
    # Compared as plain numbers, so a failure does not print transcript text.
    questions = int(passages["question"].nunique())
    assert questions > 1

    data = ROOT / "data" / "fictitious"
    themes = [t.short_name for t in load_themes(data / example["codebook"])]
    assert len(themes) == 5
    human = pd.read_csv(data / example["human"])
    assert len(human) == 20 and human["passage_id"].between(1, 26).all()
    assert set(human["theme"]) <= {*themes, settings["off_topic"]}
    roster = (folder / "roster.txt").read_text("utf-8").split("\n")
    assert len([name for name in roster if name.strip()]) >= 5


def test_placeholders_come_in_the_transcript_language():
    from src.deidentify import PLACEHOLDERS, deidentify_gliner

    class FakeModel:
        def extract_entities(self, text, labels, **options):  # noqa: D102
            start = text.index("Aminata")
            return {
                "entities": {
                    "person": [{"start": start, "end": start + 7, "confidence": 0.9}]
                }
            }

    cleaned, _ = deidentify_gliner("Merci Aminata.", FakeModel(), placeholders="fr")
    assert cleaned == "Merci <PERSONNE>."
    assert set(PLACEHOLDERS) == {"es", "en", "fr"}
