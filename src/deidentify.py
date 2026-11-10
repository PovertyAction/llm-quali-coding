"""Replace names, phone numbers, emails, and places in Spanish text with Presidio.

Runs on the laptop: spaCy's Spanish model finds names and places, Presidio's
pattern recognizers find phone numbers and emails, and an optional name list
(for example, the focus group roster) catches names the model misses.

Detectors miss things. Read the findings table before anything leaves the
laptop.
"""

from __future__ import annotations

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
