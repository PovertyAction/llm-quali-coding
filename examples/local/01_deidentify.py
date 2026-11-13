"""Step 1: split the transcript into passages and de-identify them on the laptop.

Writes:
- outputs/local/01_passages_deid.csv: one row per participant turn, with the
  moderator's question; names and numbers replaced in both. This is the only
  file that goes to Claude.
- outputs/local/01_findings.csv: what the detector replaced. It holds the
  original names, so it stays on the laptop.

Uses Presidio by default. Pass --gliner to use GLiNER-PII instead; it writes
01_passages_deid_gliner.csv and 01_findings_gliner.csv so you can compare the
two detectors. Step 2 reads the Presidio file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.chunking import parse_speakers
from src.deidentify import build_analyzer, deidentify, deidentify_gliner, load_gliner

DATA = Path("data/fictitious")
OUT = Path("outputs/local")


def main() -> None:
    """De-identify each participant turn and save passages and findings."""
    turns = parse_speakers((DATA / "grupo_focal_ficticio.md").read_text("utf-8"))
    names = (DATA / "lista_participantes.txt").read_text("utf-8").splitlines()
    names = [n for n in names if n.strip()]
    if "--gliner" in sys.argv:
        detector, model = "GLiNER-PII", load_gliner()

        def clean(text: str):
            return deidentify_gliner(text, model, known_names=names)

    else:
        detector, analyzer = "Presidio", build_analyzer(known_names=names)

        def clean(text: str):
            return deidentify(text, analyzer)

    passages, findings = [], []
    question = ""
    # A speaker label can be a real name ("Rosa Méndez:"), so each speaker is
    # exported as a numbered participant, in order of first appearance.
    speaker_ids: dict[str, str] = {}
    for turn in turns:
        if "MODERADOR" in turn["speaker"].upper():
            # The question travels with each passage, so it is cleaned too.
            question, found = clean(turn["text"])
            findings += [
                {"passage_id": len(passages) + 1, "column": "question", **f}
                for f in found
            ]
            continue
        passage_id = len(passages) + 1
        text, found = clean(turn["text"])
        passages.append(
            {
                "passage_id": passage_id,
                "speaker": speaker_ids.setdefault(
                    turn["speaker"], f"PARTICIPANTE {len(speaker_ids) + 1}"
                ),
                "question": question,
                "text": text,
            }
        )
        findings += [{"passage_id": passage_id, "column": "text", **f} for f in found]

    suffix = "_gliner" if detector == "GLiNER-PII" else ""
    passages_csv = OUT / f"01_passages_deid{suffix}.csv"
    findings_csv = OUT / f"01_findings{suffix}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(passages).to_csv(passages_csv, index=False)
    pd.DataFrame(findings).to_csv(findings_csv, index=False)
    print(f"{len(passages)} passages, {len(findings)} items replaced.")
    print(f"Wrote {passages_csv} and {findings_csv}")
    print(f"{findings_csv.name} lists only what {detector} found.")
    print(f"To see what it missed, read {passages_csv.name} next to the transcript.")


if __name__ == "__main__":
    main()
