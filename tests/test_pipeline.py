"""Checks for src/pipeline.py, using an invented transcript and a fake model."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src import pipeline
from src.coding import load_themes, save_themes, theme_prompt_text, themes_from_data

TRANSCRIPT = """# Grupo de prueba

MODERADOR: ¿Cómo les va con el ahorro?
ANA LOPEZ: Bien, ya junté para la escuela.
BETO: A mí me cuesta, vendo poco.

MODERADOR: ¿Y las reglas?
ANA LOPEZ: La multa es justa. (risas)
"""
THEMES = themes_from_data(
    [
        "Uso del ahorro: Para qué usan lo ahorrado.",
        {"name": "Reglas", "definition": "Multas y horarios.", "exclude": "Chismes"},
    ]
)
SETTINGS = {**pipeline.DEFAULTS, "workers": 2, "off_topic": "Fuera de tema"}


def test_codebook_reads_both_formats_and_round_trips(tmp_path):
    assert [t.short_name for t in THEMES] == ["Uso del ahorro", "Reglas"]
    assert theme_prompt_text(THEMES[0]) == THEMES[0].full_definition
    assert "Exclude when: Chismes" in theme_prompt_text(THEMES[1])
    save_themes(tmp_path / "c.json", THEMES)
    assert load_themes(tmp_path / "c.json") == THEMES
    # Without extra guidance the file keeps the original list-of-strings format.
    save_themes(tmp_path / "plain.json", THEMES[:1])
    assert json.loads((tmp_path / "plain.json").read_text("utf-8")) == [
        "Uso del ahorro: Para qué usan lo ahorrado."
    ]


def test_split_by_turn_renames_speakers_and_carries_the_question():
    df = pipeline.split_passages(TRANSCRIPT, participant_label="P")
    assert df["speaker"].tolist() == ["P 1", "P 2", "P 1"]
    assert df["question_id"].tolist() == [1, 1, 2]
    assert df["question"].iloc[2] == "¿Y las reglas?"
    assert "ANA" not in df.to_csv()


def test_split_by_question_fixed_size_and_no_speakers():
    by_question = pipeline.split_passages(TRANSCRIPT, unit="question")
    assert len(by_question) == 2
    assert by_question["text"].iloc[0].count("PARTICIPANT") == 2
    assert len(pipeline.split_passages("x" * 1500, unit="fixed")) == 3
    assert len(pipeline.split_passages("Uno.\n\nDos.")) == 1  # paragraphs, merged


def test_deidentify_cleans_text_and_each_question_once():
    seen = []

    def clean(text):
        seen.append(text)
        return text.replace("escuela", "<LUGAR>"), [{"entity": "LOCATION"}] * (
            "escuela" in text
        )

    passages, findings = pipeline.deidentify_passages(
        pipeline.split_passages(TRANSCRIPT), clean
    )
    assert "<LUGAR>" in passages["text"].iloc[0]
    assert len(seen) == 5  # 3 passages + 2 questions
    assert findings[["passage_id", "column"]].values.tolist() == [[1, "text"]]


def test_llm_coding_parses_good_and_bad_answers():
    answers = iter(
        [
            '```json\n{"theme": "uso del ahorro", "rationale": "School."}\n```',
            "I think this is about income.",
            '{"theme": "Fuera de tema", "rationale": "Small talk."}',
        ]
    )
    settings = {**SETTINGS, "workers": 1}
    df = pipeline.code_with_llm(
        pipeline.split_passages(TRANSCRIPT),
        THEMES,
        lambda s, u: next(answers),
        settings,
    )
    assert df["llm_theme"].tolist() == [
        "Uso del ahorro",
        pipeline.UNPARSED,
        "Fuera de tema",
    ]
    assert df["llm_rationale"].iloc[1] == "I think this is about income."


def test_llm_prompt_holds_the_study_inputs_and_multi_mode_codes_each_theme():
    prompts = []

    def complete(system, user):
        prompts.append(system)
        return "Yes." if "Reglas" in system and "multa" in user else "no"

    settings = {**SETTINGS, "coding_mode": "multi", "research_question": "¿Por qué?"}
    df = pipeline.code_with_llm(
        pipeline.split_passages(TRANSCRIPT), THEMES, complete, settings
    )
    assert df["llm: Reglas"].tolist() == ["NO", "NO", "YES"]
    assert df["llm: Uso del ahorro"].tolist() == ["NO", "NO", "NO"]
    assert "¿Por qué?" in prompts[0] and "{" not in prompts[0]
    assert any("Exclude when: Chismes" in p for p in prompts)


def test_themes_nonverbal_and_translate():
    wrapped = '{"themes": [{"name": "Ahorro", "definition": "d", "examples": "e"}]}'
    assert pipeline.discover_themes("t", lambda s, u: wrapped, SETTINGS) == [
        {"name": "Ahorro", "definition": "d", "examples": "e"}
    ]
    bad = pipeline.discover_themes("t", lambda s, u: "no themes", SETTINGS)
    assert bad[0]["name"] == pipeline.UNPARSED

    passages = pipeline.split_passages(TRANSCRIPT)
    cues = pipeline.code_nonverbal(
        passages,
        lambda s, u: (
            '{"any_cues": "yes", "cue_type": "Laughter"}' if "risas" in u else "?"
        ),
        SETTINGS,
    )
    assert cues["any_cues"].tolist() == ["NO", "NO", "YES"]

    english = pipeline.translate(passages, lambda s, u: u.upper(), SETTINGS)
    assert english["text"].iloc[1] == "A MÍ ME CUESTA, VENDO POCO."
    assert english["text_original"].iloc[1] == "A mí me cuesta, vendo poco."
    assert english["question"].iloc[0] == "¿CÓMO LES VA CON EL AHORRO?"


def fake_embed(texts, query=False):
    """Two dimensions: how much a text is about saving, and about rules."""
    return [
        [float("ahorr" in t.lower() or "escuela" in t), float("ulta" in t)]
        for t in texts
    ]


def test_embedding_coding_and_agreement():
    df = pipeline.code_with_embeddings(
        pipeline.split_passages(TRANSCRIPT), THEMES, fake_embed, 0.5, "Fuera de tema"
    )
    assert df["embedding_theme"].tolist() == [
        "Uso del ahorro",
        "Fuera de tema",
        "Reglas",
    ]
    human = pd.Series(["Uso del ahorro", "Uso del ahorro", "Reglas"], index=df.index)
    result = pipeline.agreement(human, df["embedding_theme"])
    assert result["agreement"] == pytest.approx(2 / 3)
    assert result["disagreements"] == [1]
    table = pipeline.threshold_table(human, df, "Fuera de tema")
    assert table["threshold"].tolist() == [0.0, 1.0]
    assert table["agreement"].iloc[0] == 1.0  # at 0.0 nothing is dropped


def test_batch_resumes_from_cache_and_survives_a_bad_file(tmp_path, monkeypatch):
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "a.md").write_text(TRANSCRIPT, "utf-8")
    (tmp_path / "transcripts" / "empty.txt").write_text("", "utf-8")
    (tmp_path / "roster.txt").write_text("Ana Lopez", "utf-8")
    settings = {**SETTINGS, "themes": [{"name": "Reglas", "definition": "Multas."}]}
    pipeline.save_settings(tmp_path, settings)
    assert pipeline.load_settings(tmp_path) == settings

    errors = pipeline.prepare(tmp_path, settings, pipeline.make_cleaner("none"))
    assert list(errors) == ["empty.txt"]

    # Prepared and unchanged: left alone. A changed setting: prepared again.
    assert pipeline.project_status(tmp_path, settings)["prepared"] == 1
    stamp = tmp_path / "results" / "a.md" / "passages.csv"
    before = stamp.stat().st_mtime_ns
    pipeline.prepare(tmp_path, settings, pipeline.make_cleaner("none"))
    assert stamp.stat().st_mtime_ns == before
    changed = {**settings, "unit": "question"}
    assert pipeline.project_status(tmp_path, changed)["stale"] == 1
    assert pipeline.needs_prepare(tmp_path, changed, tmp_path / "transcripts" / "a.md")
    status = pipeline.project_status(tmp_path, settings)
    assert pipeline.estimate_requests(status, settings, ["llm", "themes"]) == 3 + 1

    calls = []

    def fake_api(client, wire, model, messages, **kwargs):
        calls.append(messages)
        return '{"theme": "Reglas", "rationale": "r"}', 10, 2

    monkeypatch.setattr(pipeline, "complete_with_usage", fake_api)
    client = SimpleNamespace(base_url="http://fake")

    def run():
        complete = pipeline.Completer(client, "openai", "m", tmp_path / "cache.jsonl")
        errors = pipeline.run_batch(
            tmp_path, settings, ["embeddings", "llm"], complete, fake_embed
        )
        return complete, errors

    first, errors = run()
    assert errors == {}
    assert (first.calls, first.reused, first.input_tokens) == (3, 0, 30)
    second, _ = run()
    assert (second.calls, second.reused) == (0, 3)
    assert len(calls) == 3

    coded = pipeline.load_coded(tmp_path)
    assert coded["transcript"].unique().tolist() == ["a.md"]
    assert {"embedding_theme", "llm_theme", "relevance"} <= set(coded.columns)

    import io
    import zipfile

    names = zipfile.ZipFile(io.BytesIO(pipeline.export_zip(tmp_path))).namelist()
    assert "results/a.md/coded.csv" in names and "project.json" in names
    assert not any(
        "findings" in n or "roster" in n or "transcripts" in n for n in names
    )


def test_both_api_formats_return_text_and_token_counts():
    from src.llm_client import BACKENDS, complete_with_usage, make_client

    ns = SimpleNamespace
    sent = {}

    def create(**request):
        sent.update(request)
        return ns(
            stop_reason="end_turn",
            # Newer Claude models put a thinking block before the answer.
            content=[ns(type="thinking", thinking=""), ns(type="text", text="OK")],
            usage=ns(input_tokens=5, output_tokens=2, cache_read_input_tokens=40),
        )

    claude = ns(messages=ns(create=create))
    messages = [
        {"role": "system", "content": "codebook"},
        {"role": "user", "content": "passage"},
    ]
    assert complete_with_usage(claude, "anthropic", "m", messages) == ("OK", 45, 2)
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["messages"] == messages[1:]

    reply = ns(
        choices=[ns(message=ns(content="OK"))],
        usage=None,  # local servers
    )
    other = ns(chat=ns(completions=ns(create=lambda **request: reply)))
    assert complete_with_usage(other, "openai", "m", messages) == ("OK", 0, 0)

    for backend, wire in BACKENDS.items():
        options = {"api_key": "k", "azure_endpoint": "https://x.openai.azure.com"}
        options |= {"api_version": "2024-10-21", "aws_region": "us-east-1"}
        assert make_client(backend, **options)[1] == wire


def test_gliner_replaces_only_identifying_information():
    from src.deidentify import deidentify_gliner

    text = (
        "Mi esposo y Rosa viven en Tlacolula, México. Tel 55 1234 5678. ሰላም አበበ. I am"
    )

    class FakeModel:
        def extract_entities(self, text, labels, **options):
            def spans(*words):
                return [
                    {"start": text.index(w), "end": text.index(w) + len(w)}
                    | {"confidence": 0.99}
                    for w in words
                ]

            return {
                "entities": {
                    "person": spans("esposo", "Rosa", "አበበ", "I"),
                    "location": spans("Tlacolula", "México"),
                    "phone_number": spans("55 1234 5678"),
                }
            }

    cleaned, findings = deidentify_gliner(text, FakeModel(), keep=["méxico"])
    assert cleaned == (
        "Mi esposo y <PERSONA> viven en <LUGAR>, México. Tel <TELEFONO>. ሰላም <PERSONA>. I am"
    )
    assert [f["found"] for f in findings] == [
        "Rosa",
        "Tlacolula",
        "55 1234 5678",
        "አበበ",
    ]
    # A transcript typed without capitals: turn the filter off, use the roster.
    lower, _ = deidentify_gliner(
        "mi esposo y rosa", FakeModel2(), proper_nouns_only=False
    )
    assert lower == "mi <PERSONA> y <PERSONA>"
    roster, _ = deidentify_gliner("mi esposo y rosa", FakeModel2(), ["Rosa"])
    assert roster == "mi esposo y <PERSONA>"


class FakeModel2:
    """Tag "esposo" and "rosa" as people, as GLiNER2-PII does."""

    def extract_entities(self, text, labels, **options):  # noqa: D102
        return {
            "entities": {
                "person": [
                    {"start": text.index(w), "end": text.index(w) + len(w)}
                    | {"confidence": 0.99}
                    for w in ("esposo", "rosa")
                ]
            }
        }


def test_reviewed_mark_and_outdated_codes(tmp_path):
    (tmp_path / "transcripts").mkdir()
    file = tmp_path / "transcripts" / "a.md"
    file.write_text(TRANSCRIPT, "utf-8")
    pipeline.prepare(tmp_path, SETTINGS, pipeline.make_cleaner("none"))
    assert not pipeline.is_reviewed(tmp_path, file)
    pipeline.set_reviewed(tmp_path, file, True)
    assert pipeline.project_status(tmp_path, SETTINGS)["reviewed"] == 1
    # Preparing again changes the passages, so the mark no longer holds.
    pipeline.prepare(tmp_path, SETTINGS, pipeline.make_cleaner("none"), force=True)
    assert not pipeline.is_reviewed(tmp_path, file)

    before = pipeline.coding_fingerprint(SETTINGS)
    assert pipeline.coding_fingerprint({**SETTINGS, "workers": 9}) == before
    assert pipeline.coding_fingerprint({**SETTINGS, "threshold": 0.5}) != before

    seen = []
    errors = pipeline.run_batch(
        tmp_path,
        {**SETTINGS, "themes": [{"name": "Reglas", "definition": "d"}]},
        ["llm"],
        lambda s, u: 1 / 0,
        on_progress=lambda f, t: seen.append(f),
        explain=lambda exc: f"explained {type(exc).__name__}",
    )
    assert errors == {"a.md": "explained ZeroDivisionError"}
    assert seen == sorted(seen)  # the bar never goes backwards
