"""Step 4: compare model codes with a person's codes on the same passages.

Reads the human sample (data/fictitious/codigos_humanos.csv), the embedding
codes (outputs/local/02_coded.csv), and Claude's codes if they exist
(outputs/local/03_claude_codes.csv, columns passage_id and theme).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

OUT = Path("outputs/local")
OFF_TOPIC = "Fuera de tema"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def report(name: str, human: pd.Series, model: pd.Series, text: pd.Series) -> None:
    """Print agreement, Cohen's kappa, a confusion table, and disagreements."""
    print(f"\n=== {name} vs human, {len(human)} passages ===")
    print(f"Agreement: {(human == model).mean():.0%}")
    print(f"Cohen's kappa: {cohen_kappa_score(human, model):.2f}")
    print(pd.crosstab(human, model, rownames=["human"], colnames=[name]))
    for pid in human.index[human != model]:
        print(f"- {pid}: human {human[pid]!r}, {name} {model[pid]!r}: {text[pid]}")


def threshold_table(human: pd.Series, coded: pd.DataFrame) -> None:
    """Print agreement with the person for each candidate THRESHOLD in step 2."""
    print("\n=== embeddings: agreement by THRESHOLD (step 2) ===")
    print("Pick the best row, then go a little lower: a passage dropped by")
    print("mistake is harder to notice than small talk kept by mistake.")
    for t in sorted(set(coded["relevance"])):
        model = coded["nearest_theme"].where(coded["relevance"] >= t, OFF_TOPIC)
        kappa = cohen_kappa_score(human, model)
        print(f"  {t:.2f}  agreement {(human == model).mean():4.0%}  kappa {kappa:.2f}")


def main() -> None:
    """Report each model's codes against the human sample."""
    coded_path = OUT / "02_coded.csv"
    if not coded_path.exists():
        raise SystemExit(f"Missing {coded_path}. Run step 2: just local-classify")
    human = pd.read_csv("data/fictitious/codigos_humanos.csv", index_col="passage_id")
    coded = pd.read_csv(coded_path, index_col="passage_id").loc[human.index]
    report("embeddings", human["theme"], coded["embedding_theme"], coded["text"])
    threshold_table(human["theme"], coded)

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
