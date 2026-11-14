"""Replace names, phone numbers, emails, and places in Spanish text with Presidio.

Runs on the laptop: spaCy's Spanish model finds names and places, Presidio's
pattern recognizers find phone numbers and emails, and an optional name list
(for example, the focus group roster) catches names the model misses.

Detectors miss things, and the findings table lists only what was found.
Before anything leaves the laptop, read the de-identified text next to the
original to look for names and numbers that are still there.
"""

from __future__ import annotations

import re

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

LANGUAGE = "es"
ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]
LABELS = {
    "PERSON": "<PERSONA>",
    "PHONE_NUMBER": "<TELEFONO>",
    "EMAIL_ADDRESS": "<CORREO>",
    "LOCATION": "<LUGAR>",
}


def build_analyzer(known_names: list[str] | None = None) -> AnalyzerEngine:
    """Build a Spanish analyzer, adding a recognizer for a list of known names."""
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": LANGUAGE, "model_name": "es_core_news_md"}],
        }
    ).create_engine()
    registry = RecognizerRegistry(supported_languages=[LANGUAGE])
    registry.add_recognizer(SpacyRecognizer(supported_language=LANGUAGE))
    registry.add_recognizer(EmailRecognizer(supported_language=LANGUAGE))
    registry.add_recognizer(
        PhoneRecognizer(supported_language=LANGUAGE, supported_regions=("MX", "US"))
    )
    if known_names:
        registry.add_recognizer(
            PatternRecognizer(
                supported_entity="PERSON",
                supported_language=LANGUAGE,
                deny_list=known_names,
                name="KnownNamesRecognizer",
            )
        )
    return AnalyzerEngine(
        nlp_engine=nlp_engine, registry=registry, supported_languages=[LANGUAGE]
    )


def _identifies(found: str, keep: set[str], proper_nouns_only: bool = False) -> bool:
    """Decide whether a detected person or place should be replaced.

    - `keep` holds lowercased words the researcher wants left in the text: a
      country or large city does not identify anyone, and a detector cannot
      tell a village from a capital.
    - With `proper_nouns_only`, a span written entirely in lowercase is left
      alone. GLiNER2-PII tags kin and role words ("esposo", "tesorera") and
      generic places ("casa", "mercado") with the same confidence as real
      names, so the score cannot separate them; the capital letter can.
      Scripts without capital letters (Amharic, Arabic, Hindi) are not
      affected, because `islower()` is False for them.
    """
    if found.strip().casefold() in keep:
        return False
    if len(found.strip()) < 2:
        return False  # the English pronoun "I" is capitalized but is no name
    return not (proper_nouns_only and found.islower())


# Placeholders in the language of the transcript, so the cleaned text reads
# naturally. Keys are the GLiNER labels; KINDS maps Presidio's names to them.
PLACEHOLDERS = {
    "es": {
        "person": "<PERSONA>",
        "phone_number": "<TELEFONO>",
        "email": "<CORREO>",
        "location": "<LUGAR>",
    },
    "en": {
        "person": "<PERSON>",
        "phone_number": "<PHONE>",
        "email": "<EMAIL>",
        "location": "<PLACE>",
    },
    "fr": {
        "person": "<PERSONNE>",
        "phone_number": "<TELEPHONE>",
        "email": "<COURRIEL>",
        "location": "<LIEU>",
    },
}
KINDS = {
    "PERSON": "person",
    "PHONE_NUMBER": "phone_number",
    "EMAIL_ADDRESS": "email",
    "LOCATION": "location",
}


def deidentify(
    text: str,
    analyzer: AnalyzerEngine,
    keep: list[str] | None = None,
    placeholders: str = "es",
) -> tuple[str, list[dict[str, object]]]:
    """Return the text with entities replaced, and a list of what was found.

    Names and places listed in `keep` are left in the text. `placeholders` is
    the language of the replacement tags (a key of PLACEHOLDERS).
    """
    tags = PLACEHOLDERS[placeholders]
    kept = {k.strip().casefold() for k in keep or []}
    results = [
        r
        for r in analyzer.analyze(text=text, entities=ENTITIES, language=LANGUAGE)
        if r.entity_type not in ("PERSON", "LOCATION")
        or _identifies(text[r.start : r.end], kept)
    ]
    findings = [
        {
            "entity": r.entity_type,
            "found": text[r.start : r.end],
            "score": round(r.score, 2),
            "recognizer": r.recognition_metadata.get("recognizer_name", ""),
        }
        for r in results
    ]
    operators = {
        entity: OperatorConfig("replace", {"new_value": tags[kind]})
        for entity, kind in KINDS.items()
    }
    anonymized = AnonymizerEngine().anonymize(
        text=text, analyzer_results=results, operators=operators
    )
    return anonymized.text, findings


# GLiNER2-PII: a second detector, a small multilingual transformer (Spanish is
# on its model card) that reads the whole sentence instead of matching
# patterns. It finds names without a list. It also tags lowercase words like
# "esposo" and "casa", which deidentify_gliner() leaves alone by default.
GLINER_MODEL = "fastino/gliner2-privacy-filter-PII-multi"
GLINER_LABELS = {
    "person": "<PERSONA>",
    "phone_number": "<TELEFONO>",
    "email": "<CORREO>",
    "location": "<LUGAR>",
}


def load_gliner():
    """Load GLiNER2-PII (downloaded once from Hugging Face, then runs offline)."""
    from gliner2 import GLiNER2

    return GLiNER2.from_pretrained(GLINER_MODEL)


def deidentify_gliner(
    text: str,
    model,
    known_names: list[str] | None = None,
    threshold: float = 0.5,
    keep: list[str] | None = None,
    proper_nouns_only: bool = True,
    placeholders: str = "es",
) -> tuple[str, list[dict[str, object]]]:
    """Replace entities with GLiNER2-PII and the name list; same output as deidentify().

    Only identifying information is replaced: names of people and places,
    phone numbers and email addresses. See `_identifies` for `keep` and
    `proper_nouns_only`. Set `proper_nouns_only=False` for a transcript typed
    without capital letters, where a name looks like any other word. Names in
    `known_names` are always replaced.
    """
    kept = {k.strip().casefold() for k in keep or []}
    result = model.extract_entities(
        text,
        list(GLINER_LABELS),
        threshold=threshold,
        include_confidence=True,
        include_spans=True,
    )
    spans = [
        (e["start"], e["end"], label, round(e["confidence"], 2), "GLiNER2-PII")
        for label, found in result["entities"].items()
        for e in found
        if label not in ("person", "location")
        or _identifies(text[e["start"] : e["end"]], kept, proper_nouns_only)
    ]
    for name in known_names or []:
        for m in re.finditer(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            spans.append((m.start(), m.end(), "person", 1.0, "KnownNames"))
    # Merge overlapping spans so the whole union is replaced. Otherwise "Rosa"
    # (model) inside "Rosa Méndez" (name list) would leave "Méndez" behind.
    # The union takes the label of its widest span, so a name found inside an
    # email address is still reported as an email.
    merged: list[list] = []  # [start, end, widest span]
    for span in sorted(spans):
        if merged and span[0] < merged[-1][1]:
            group = merged[-1]
            group[1] = max(group[1], span[1])
            group[2] = max(group[2], span, key=lambda s: s[1] - s[0])
        else:
            merged.append([span[0], span[1], span])
    findings, out, last = [], [], 0
    for start, end, (_, _, label, score, source) in merged:
        out += [text[last:start], PLACEHOLDERS[placeholders][label]]
        last = end
        findings.append(
            {
                "entity": label,
                "found": text[start:end],
                "score": score,
                "recognizer": source,
            }
        )
    return "".join(out) + text[last:], findings
