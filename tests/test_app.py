"""Every page of the web app renders for the example project without an error."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP))
PAGES = ["project", "study", "transcripts", "run", "results", "human"]


@pytest.fixture
def example(tmp_path, monkeypatch):
    """Create the example project, prepared without a detector, in a temp folder."""
    monkeypatch.setenv("QUALI_PROJECTS_DIR", str(tmp_path))
    import main

    from src import pipeline as pl

    main.make_example("es")
    folder = tmp_path / "example"
    settings = pl.load_settings(folder)
    assert pl.prepare(folder, settings, pl.make_cleaner("none")) == {}

    def complete(system, user):
        if user.startswith("TRANSCRIPT"):
            return '[{"name": "Ahorro", "definition": "d", "examples": "e"}]'
        return '{"theme": "Reglas y multas", "rationale": "r"}'

    steps = ["embeddings", "llm", "llm2", "themes"]
    embed = lambda texts, query=False: [[1.0, float(len(t) % 2)] for t in texts]  # noqa: E731
    assert pl.run_batch(folder, settings, steps, complete, embed, complete) == {}
    return folder


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(example, page):
    at = AppTest.from_string(f"import main\nmain.{page}_page()", default_timeout=60)
    at.session_state["project"] = "example"
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value


def test_pages_ask_for_a_project_when_none_is_open(example):
    at = AppTest.from_string("import main\nmain.study_page()").run()
    assert not at.exception
    assert "No project is open" in at.info[0].value


def test_shell_renders_with_the_data_destination(example):
    at = AppTest.from_file(str(APP / "main.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    # The default model server is on this computer, so nothing leaves it.
    assert "runs on this computer" in at.sidebar.success[0].value


def test_results_show_yes_no_coding(example):
    from src import pipeline as pl

    settings = {**pl.load_settings(example), "coding_mode": "multi"}
    pl.save_settings(example, settings)
    assert pl.run_batch(example, settings, ["llm"], lambda s, u: "YES") == {}
    at = AppTest.from_string("import main\nmain.results_page()", default_timeout=60)
    at.session_state["project"] = "example"
    at.run()
    assert not at.exception, at.exception
    assert "Passages per theme" in [h.value for h in at.header]
    # One line per passage and per set of codes, each naming every theme.
    assert any("Reglas y multas" in m.value for m in at.markdown)


def test_approval_follows_the_least_approved_destination():
    import main

    from src import pipeline as pl

    local = dict(pl.DEFAULTS)
    assert main.cloud_destinations(local) == []
    claude = {**local, "backend": "anthropic"}
    assert main.approval(main.cloud_destinations(claude)) == "Internal"
    mixed = {**local, "backend": "azure", "embed_backend": "openai"}
    assert main.approval(main.cloud_destinations(mixed)) is None
    assert main.cloud_destinations(mixed, ["embeddings"]) == [("openai", "OpenAI")]
    remote = {**local, "base_url": "http://10.0.0.5:11434/v1"}
    assert main.cloud_destinations(remote)[0][0] == "remote"
    assert main.approval(main.cloud_destinations(remote)) is None
