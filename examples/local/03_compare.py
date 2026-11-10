"""Step 3: compare model codes with a person's codes on the same passages.

Reads the human sample (data/fictitious/codigos_humanos.csv), the embedding
codes (outputs/local/02_coded.csv), and Claude's codes if they exist
(outputs/local/claude_codes.csv, columns passage_id and theme).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

OUT = Path("outputs/local")
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


def main() -> None:
    """Report each model's codes against the human sample."""
    human = pd.read_csv("data/fictitious/codigos_humanos.csv", index_col="passage_id")
    coded = pd.read_csv(OUT / "02_coded.csv", index_col="passage_id").loc[human.index]
    report("embeddings", human["theme"], coded["embedding_theme"], coded["text"])

    claude_path = OUT / "claude_codes.csv"
    if claude_path.exists():
        claude = pd.read_csv(claude_path, index_col="passage_id").loc[human.index]
        report("claude", human["theme"], claude["theme"], coded["text"])
    else:
        print(f"\nNo {claude_path} yet. Ask Claude Code to write it, then rerun.")


if __name__ == "__main__":
    main()
