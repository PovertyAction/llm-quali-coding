"""Step 1: split the transcript into passages and de-identify them on the laptop.

Writes:
- outputs/local/01_passages_deid.csv: one row per participant turn, names and
  numbers replaced. This is the only file that goes to Claude.
- outputs/local/01_findings.csv: what Presidio replaced. It holds the original
  names, so it stays on the laptop.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.chunking import parse_speakers
from src.deidentify import build_analyzer, deidentify

DATA = Path("data/fictitious")
OUT = Path("outputs/local")


def main() -> None:
    """De-identify each participant turn and save passages and findings."""
    turns = parse_speakers((DATA / "grupo_focal_ficticio.md").read_text("utf-8"))
    names = (DATA / "lista_participantes.txt").read_text("utf-8").splitlines()
    analyzer = build_analyzer(known_names=[n for n in names if n.strip()])

    passages, findings = [], []
    question = ""
    for turn in turns:
        if "MODERADOR" in turn["speaker"].upper():
            question = turn["text"]
            continue
        passage_id = len(passages) + 1
        text, found = deidentify(turn["text"], analyzer)
        passages.append(
            {
                "passage_id": passage_id,
                "speaker": turn["speaker"],
                "question": question,
                "text": text,
            }
        )
        findings += [{"passage_id": passage_id, **f} for f in found]

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(passages).to_csv(OUT / "01_passages_deid.csv", index=False)
    pd.DataFrame(findings).to_csv(OUT / "01_findings.csv", index=False)
    print(f"{len(passages)} passages, {len(findings)} items replaced.")
    print(f"Wrote {OUT / '01_passages_deid.csv'} and {OUT / '01_findings.csv'}")
    print("Read 01_findings.csv: what did Presidio miss?")


if __name__ == "__main__":
    main()
