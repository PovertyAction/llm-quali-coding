"""Step 4: compare model codes with a person's codes on the same passages.

Reads the human sample (data/fictitious/codigos_humanos.csv), the embedding
codes (outputs/local/02_coded.csv), and Claude's codes if they exist
(outputs/local/03_claude_codes.csv, columns passage_id and theme).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline import agreement, threshold_table

OUT = Path("outputs/local")
OFF_TOPIC = "Fuera de tema"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def report(name: str, human: pd.Series, model: pd.Series, text: pd.Series) -> None:
    """Print agreement, Cohen's kappa, a confusion table, and disagreements."""
    result = agreement(human, model, "human", name)
    print(f"\n=== {name} vs human, {result['n']} passages ===")
    print(f"Agreement: {result['agreement']:.0%}")
    print(f"Cohen's kappa: {result['kappa']:.2f}")
    print(result["confusion"])
    for pid in result["disagreements"]:
        print(f"- {pid}: human {human[pid]!r}, {name} {model[pid]!r}: {text[pid]}")


def print_threshold_table(human: pd.Series, coded: pd.DataFrame) -> None:
    """Print agreement with the person for each candidate THRESHOLD in step 2."""
    print("\n=== embeddings: agreement by THRESHOLD (step 2) ===")
    print("Pick the best row, then go a little lower: a passage dropped by")
    print("mistake is harder to notice than small talk kept by mistake.")
    for row in threshold_table(human, coded, OFF_TOPIC).itertuples():
        print(
            f"  {row.threshold:.2f}  agreement {row.agreement:4.0%}"
            f"  kappa {row.kappa:.2f}"
        )


def main() -> None:
    """Report each model's codes against the human sample."""
    coded_path = OUT / "02_coded.csv"
    if not coded_path.exists():
        raise SystemExit(f"Missing {coded_path}. Run step 2: just local-classify")
    human = pd.read_csv("data/fictitious/codigos_humanos.csv", index_col="passage_id")
    coded = pd.read_csv(coded_path, index_col="passage_id").loc[human.index]
    report("embeddings", human["theme"], coded["embedding_theme"], coded["text"])
    print_threshold_table(human["theme"], coded)

    claude_path = OUT / "03_claude_codes.csv"
    if not claude_path.exists():
        print(f"\nNo {claude_path} yet. Do step 3 (ask Claude Code), then rerun.")
        return
    claude = pd.read_csv(claude_path, index_col="passage_id")
    missing = human.index.difference(claude.index).tolist()
    if missing:
        raise SystemExit(f"{claude_path} has no code for passages {missing}.")
    claude = claude.loc[human.index]
    report("claude", human["theme"], claude["theme"], coded["text"])


if __name__ == "__main__":
    main()
