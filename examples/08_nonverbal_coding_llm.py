from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.llm_tasks import code_nonverbal_cues
from src.openai_client import get_client


def main() -> None:
    """Code non-verbal cues using structured LLM output."""
    client = get_client()

    inp = Path("outputs/04_relevant_chunks.csv")
    if not inp.exists():
        raise FileNotFoundError(
            "Missing outputs/04_relevant_chunks.csv. Run step 04 first."
        )

    df = pd.read_csv(inp)
    df["embedding"] = df["embedding"].apply(json.loads)

    # For cost control in a classroom, default to a small sample
    n = min(10, len(df))
    cue_yes: list[str] = []
    cue_type: list[str] = []

    for i in range(n):
        res = code_nonverbal_cues(client, df.loc[i, "text"])
        cue_yes.append(res.get("any_nonverbal_cue", "NO"))
        cue_type.append(res.get("cue_type", ""))

    df_out = df.head(n).copy()
    df_out["any_nonverbal_cue"] = cue_yes
    df_out["cue_type"] = cue_type

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "08_nonverbal_coding.csv"

    df_out["embedding"] = df_out["embedding"].apply(json.dumps)
    df_out.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(df_out[["chunk_id", "any_nonverbal_cue", "cue_type"]])


if __name__ == "__main__":
    main()
