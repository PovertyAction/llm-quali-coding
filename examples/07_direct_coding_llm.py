from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.llm_tasks import code_chunk_yes_no
from src.openai_client import get_client


def main() -> None:
    """Apply direct coding to chunks using LLM."""
    client = get_client()

    inp = Path("outputs/04_relevant_chunks.csv")
    if not inp.exists():
        raise FileNotFoundError(
            "Missing outputs/04_relevant_chunks.csv. Run step 04 first."
        )

    df = pd.read_csv(inp)

    # Pick one theme to code for (demo). You can replace this with your own.
    theme = "Joint debrief-and-adjust loops: Teams met after sessions to review what worked/what didn’t and co-adjust the next session, improving flow and fit over time."

    n = min(10, len(df))
    codes = []
    for i in range(n):
        codes.append(code_chunk_yes_no(client, df.loc[i, "text"], theme))

    df_out = df.head(n).copy()
    df_out["llm_code_yes_no"] = codes

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "07_direct_coding_llm.csv"
    df_out.to_csv(out_path, index=False)

    print(f"Coded {n} chunks.")
    print(f"Wrote: {out_path}")
    print(df_out["llm_code_yes_no"].value_counts())


if __name__ == "__main__":
    main()
