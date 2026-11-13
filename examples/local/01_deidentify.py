"""Step 1: split the transcript into passages and de-identify them on the laptop.

Writes:
- outputs/local/01_passages_deid.csv: one row per participant turn, names and
  numbers replaced. This is the only file that goes to Claude.
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
    for turn in turns:
        if "MODERADOR" in turn["speaker"].upper():
            question = turn["text"]
            continue
        passage_id = len(passages) + 1
        text, found = clean(turn["text"])
        passages.append(
            {
                "passage_id": passage_id,
                "speaker": turn["speaker"],
                "question": question,
                "text": text,
            }
        )
        findings += [{"passage_id": passage_id, **f} for f in found]

    suffix = "_gliner" if detector == "GLiNER-PII" else ""
    passages_csv = OUT / f"01_passages_deid{suffix}.csv"
    findings_csv = OUT / f"01_findings{suffix}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(passages).to_csv(passages_csv, index=False)
    pd.DataFrame(findings).to_csv(findings_csv, index=False)
    print(f"{len(passages)} passages, {len(findings)} items replaced.")
    print(f"Wrote {passages_csv} and {findings_csv}")
    print(f"Read {findings_csv.name}: what did {detector} miss?")


if __name__ == "__main__":
    main()
