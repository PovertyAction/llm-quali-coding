"""Step 1: split the transcript into passages and de-identify them on the laptop.

Writes:
- outputs/local/01_passages_deid.csv: one row per participant turn, with the
  moderator's question; names and numbers replaced in both. This is the only
  file that goes to Claude.
- outputs/local/01_findings.csv: what the detector replaced. It holds the
  original names, so it stays on the laptop.

Uses Presidio by default. Pass --gliner to use GLiNER2-PII instead; it writes
01_passages_deid_gliner.csv and 01_findings_gliner.csv so you can compare the
two detectors. Step 2 reads the Presidio file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.pipeline import deidentify_passages, make_cleaner, split_passages

DATA = Path("data/fictitious")
OUT = Path("outputs/local")


def main() -> None:
    """De-identify each participant turn and save passages and findings."""
    names = (DATA / "lista_participantes.txt").read_text("utf-8").splitlines()
    names = [n for n in names if n.strip()]
    gliner = "--gliner" in sys.argv
    detector = "GLiNER2-PII" if gliner else "Presidio"
    clean = make_cleaner("gliner" if gliner else "presidio", names)

    # One passage per participant turn, with the moderator's question. A
    # speaker label can be a real name ("Rosa Méndez:"), so each speaker is
    # exported as a numbered participant, in order of first appearance.
    passages = split_passages(
        (DATA / "grupo_focal_ficticio.md").read_text("utf-8"),
        moderator_labels=["MODERADOR"],
        participant_label="PARTICIPANTE",
    )
    # The question travels with each passage, so it is cleaned too.
    passages, findings = deidentify_passages(passages, clean)
    passages = passages[["passage_id", "speaker", "question", "text"]]

    suffix = "_gliner" if gliner else ""
    passages_csv = OUT / f"01_passages_deid{suffix}.csv"
    findings_csv = OUT / f"01_findings{suffix}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    passages.to_csv(passages_csv, index=False)
    findings.to_csv(findings_csv, index=False)
    print(f"{len(passages)} passages, {len(findings)} items replaced.")
    print(f"Wrote {passages_csv} and {findings_csv}")
    print(f"{findings_csv.name} lists only what {detector} found.")
    print(f"To see what it missed, read {passages_csv.name} next to the transcript.")


if __name__ == "__main__":
    main()
