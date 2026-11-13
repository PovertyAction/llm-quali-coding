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


def deidentify(
    text: str, analyzer: AnalyzerEngine
) -> tuple[str, list[dict[str, object]]]:
    """Return the text with entities replaced, and a list of what was found."""
    results = analyzer.analyze(text=text, entities=ENTITIES, language=LANGUAGE)
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
        entity: OperatorConfig("replace", {"new_value": label})
        for entity, label in LABELS.items()
    }
    anonymized = AnonymizerEngine().anonymize(
        text=text, analyzer_results=results, operators=operators
    )
    return anonymized.text, findings


# GLiNER-PII: a second detector, a small transformer that reads the whole
# sentence instead of matching patterns. Its model card lists English only, so
# check its findings on Spanish text as carefully as Presidio's.
GLINER_MODEL = "nvidia/gliner-PII"
GLINER_LABELS = {
    "person": "<PERSONA>",
    "phone_number": "<TELEFONO>",
    "email": "<CORREO>",
    "location": "<LUGAR>",
}


def load_gliner():
    """Load GLiNER-PII (downloaded once from Hugging Face, then runs offline)."""
    from gliner import GLiNER

    return GLiNER.from_pretrained(GLINER_MODEL)


def deidentify_gliner(
    text: str,
    model,
    known_names: list[str] | None = None,
    threshold: float = 0.3,
) -> tuple[str, list[dict[str, object]]]:
    """Replace entities with GLiNER-PII and the name list; same output as deidentify()."""
    spans = [
        (e["start"], e["end"], e["label"], round(e["score"], 2), "GLiNER-PII")
        for e in model.predict_entities(text, list(GLINER_LABELS), threshold=threshold)
    ]
    for name in known_names or []:
        for m in re.finditer(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            spans.append((m.start(), m.end(), "person", 1.0, "KnownNames"))
    # Merge overlapping spans so the whole union is replaced. Otherwise "Rosa"
    # (model) inside "Rosa Méndez" (name list) would leave "Méndez" behind.
    merged: list[tuple] = []
    for span in sorted(spans):
        if merged and span[0] < merged[-1][1]:
            first = merged[-1]
            merged[-1] = (first[0], max(first[1], span[1]), *first[2:])
        else:
            merged.append(span)
    findings, out, last = [], [], 0
    for start, end, label, score, source in merged:
        out += [text[last:start], GLINER_LABELS[label]]
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
