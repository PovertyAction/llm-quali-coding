from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.coding import load_themes
from src.llm_tasks import code_yes_no_for_theme, extract_general_themes
from src.openai_client import get_client, get_transcript_path, load_config


def generate_comparison_html(
    df: pd.DataFrame,
    themes_a: str,
    themes_b: str,
    model_a: str,
    model_b: str,
    reference_theme: str,
    output_path: Path,
) -> None:
    """Generate side-by-side HTML comparison report."""
    col_a = f"decision_{model_a}"
    col_b = f"decision_{model_b}"

    total = len(df)
    agree = int(df["agreement"].sum())
    disagree = total - agree
    agree_pct = agree / total * 100 if total > 0 else 0

    a_yes_b_no = int(((df[col_a] == "YES") & (df[col_b] == "NO")).sum())
    a_no_b_yes = int(((df[col_a] == "NO") & (df[col_b] == "YES")).sum())

    rows_html = ""
    for _, row in df.iterrows():
        row_class = "agree" if row["agreement"] else "disagree"
        agree_label = "Yes" if row["agreement"] else "No"
        dec_a = row[col_a]
        dec_b = row[col_b]
        badge_a = (
            f'<span class="badge-{"yes" if dec_a == "YES" else "no"}">{dec_a}</span>'
        )
        badge_b = (
            f'<span class="badge-{"yes" if dec_b == "YES" else "no"}">{dec_b}</span>'
        )
        preview = str(row["text"])[:130].replace("<", "&lt;").replace(">", "&gt;")
        if len(str(row["text"])) > 130:
            preview += "…"
        rows_html += f"""
            <tr class="{row_class}">
                <td>{row["chunk_id"]}</td>
                <td>{badge_a}</td>
                <td>{badge_b}</td>
                <td>{agree_label}</td>
                <td class="chunk-preview" title="{preview}">{preview}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison - Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5; line-height: 1.6;
        }}
        .container {{
            max-width: 1400px; margin: 0 auto; background: white;
            padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #9b59b6; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stats {{
            background: #f0e6ff; padding: 15px; border-radius: 5px;
            margin: 20px 0; border-left: 4px solid #9b59b6;
        }}
        .stats-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px; margin-top: 15px;
        }}
        .stat-card {{
            background: white; padding: 15px; border-radius: 5px; text-align: center;
        }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #9b59b6; }}
        .stat-label {{ color: #7f8c8d; font-size: 0.9em; }}
        .reference-theme {{
            background: #e8f5e9; border-left: 4px solid #27ae60;
            padding: 15px; border-radius: 4px; margin: 20px 0;
        }}
        .themes-grid {{
            display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;
        }}
        .model-box {{ border: 2px solid #e0e0e0; border-radius: 8px; padding: 20px; }}
        .model-box h3 {{
            margin: 0 0 15px 0; padding: 10px; border-radius: 5px;
            color: white; font-size: 1em;
        }}
        .model-a h3 {{ background: #3498db; }}
        .model-b h3 {{ background: #e74c3c; }}
        .model-themes {{
            white-space: pre-wrap; font-size: 0.88em; color: #2c3e50;
            max-height: 420px; overflow-y: auto;
        }}
        .disagree-summary {{
            background: #fff3cd; border-left: 4px solid #f39c12;
            padding: 12px 15px; border-radius: 4px; margin: 15px 0;
            font-size: 0.9em;
        }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; margin-top: 15px; }}
        th {{
            background: #2c3e50; color: white; padding: 10px; text-align: left;
            position: sticky; top: 0;
        }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #ecf0f1; vertical-align: top; }}
        tr.agree:hover {{ background: #f0faf0; }}
        tr.disagree {{ background: #fff5f5; }}
        tr.disagree:hover {{ background: #ffe8e8; }}
        .badge-yes {{
            display: inline-block; background: #27ae60; color: white;
            padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 0.85em;
        }}
        .badge-no {{
            display: inline-block; background: #95a5a6; color: white;
            padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 0.85em;
        }}
        .chunk-preview {{ max-width: 380px; color: #555; font-size: 0.85em; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Model Comparison Report</h1>

    <div class="stats">
        <strong>Overview</strong>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total}</div>
                <div class="stat-label">Total chunks coded</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{agree}</div>
                <div class="stat-label">Chunks in agreement</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{disagree}</div>
                <div class="stat-label">Chunks in disagreement</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{agree_pct:.1f}%</div>
                <div class="stat-label">Agreement rate</div>
            </div>
        </div>
    </div>

    <div class="disagree-summary">
        <strong>Disagreement breakdown:</strong>
        &nbsp; {model_a}=YES / {model_b}=NO: <strong>{a_yes_b_no}</strong> chunks
        &nbsp;&nbsp;|&nbsp;&nbsp;
        {model_a}=NO / {model_b}=YES: <strong>{a_no_b_yes}</strong> chunks
    </div>

    <div class="reference-theme">
        <strong>Reference theme used for YES/NO coding:</strong><br>
        {reference_theme}
    </div>

    <h2>Themes Extracted by Each Model</h2>
    <div class="themes-grid">
        <div class="model-box model-a">
            <h3>{model_a}</h3>
            <div class="model-themes">{themes_a}</div>
        </div>
        <div class="model-box model-b">
            <h3>{model_b}</h3>
            <div class="model-themes">{themes_b}</div>
        </div>
    </div>

    <h2>Chunk-by-Chunk Coding Comparison</h2>
    <p>Rows highlighted in red indicate disagreements between models.</p>
    <table>
        <thead>
            <tr>
                <th>Chunk ID</th>
                <th>{model_a}</th>
                <th>{model_b}</th>
                <th>Agreement</th>
                <th>Chunk preview</th>
            </tr>
        </thead>
        <tbody>{rows_html}
        </tbody>
    </table>
</div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    """Compare two LLM models on qualitative coding tasks across all chunks."""
    client = get_client()
    cfg = load_config()

    model_a = os.getenv("COMPARE_MODEL_A", "gpt-4o-mini")
    model_b = os.getenv("COMPARE_MODEL_B", "gpt-4o")

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"  Model A: {model_a}")
    print(f"  Model B: {model_b}")

    # Load transcript
    inp = get_transcript_path(cfg.transcript_language)
    if not inp.exists():
        raise FileNotFoundError(
            f"Transcript file not found for language '{cfg.transcript_language}': {inp}\n"
            "Set TRANSCRIPT_LANGUAGE in your .env file and ensure the file exists."
        )
    transcript = inp.read_text(encoding="utf-8")
    print(f"\nTranscript: {inp}  (TRANSCRIPT_LANGUAGE={cfg.transcript_language})")

    # Load chunks
    chunks_path = Path("outputs/01_chunks_with_embeddings.csv")
    if not chunks_path.exists():
        raise FileNotFoundError(
            "Missing outputs/01_chunks_with_embeddings.csv. Run step 02 first."
        )
    df = pd.read_csv(chunks_path)
    print(f"Chunks loaded: {len(df)}")

    # Step 1: Extract themes with both models
    print(f"\n[1/3] Extracting themes with {model_a}...")
    themes_a = extract_general_themes(client, transcript, model=model_a)

    print(f"\n[2/3] Extracting themes with {model_b}...")
    themes_b = extract_general_themes(client, transcript, model=model_b)

    # Step 2: Load reference theme for YES/NO coding
    themes_file = Path("data/themes/help_themes.json")
    if themes_file.exists():
        all_themes = load_themes(themes_file)
        reference_theme = (
            all_themes[0].full_definition
            if all_themes
            else (
                "Facilitators discuss strategies, resources, or adaptations that helped "
                "integrate the program into existing family services."
            )
        )
    else:
        reference_theme = (
            "Facilitators discuss strategies, resources, or adaptations that helped "
            "integrate the program into existing family services."
        )

    print(f"\n[3/3] Running YES/NO coding on all {len(df)} chunks with both models...")
    print(f"Reference theme: {reference_theme[:90]}...")

    decisions_a: list[str] = []
    decisions_b: list[str] = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Coding chunks"):
        decisions_a.append(
            code_yes_no_for_theme(client, row["text"], reference_theme, model=model_a)
        )
        decisions_b.append(
            code_yes_no_for_theme(client, row["text"], reference_theme, model=model_b)
        )

    col_a = f"decision_{model_a}"
    col_b = f"decision_{model_b}"
    df[col_a] = decisions_a
    df[col_b] = decisions_b
    df["agreement"] = df[col_a] == df[col_b]

    # Save outputs
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Drop embedding column before saving (large, not needed in comparison output)
    cols_to_save = [c for c in df.columns if c != "embedding"]
    out_csv = out_dir / "07_model_comparison.csv"
    df[cols_to_save].to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")

    # Save raw theme extraction text
    (out_dir / "07_themes_model_a.txt").write_text(
        f"Model: {model_a}\n\n{themes_a}", encoding="utf-8"
    )
    (out_dir / "07_themes_model_b.txt").write_text(
        f"Model: {model_b}\n\n{themes_b}", encoding="utf-8"
    )

    # Generate HTML report
    html_path = out_dir / "07_model_comparison_report.html"
    generate_comparison_html(
        df, themes_a, themes_b, model_a, model_b, reference_theme, html_path
    )
    print(f"Wrote: {html_path}")

    # Console summary
    total = len(df)
    agree = int(df["agreement"].sum())
    disagree = total - agree
    agree_pct = agree / total * 100

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nTotal chunks coded:  {total}")
    print(f"Agreement:           {agree}/{total} ({agree_pct:.1f}%)")
    print(f"Disagreement:        {disagree}/{total} ({100 - agree_pct:.1f}%)")

    disagreements = df[~df["agreement"]]
    if len(disagreements) > 0:
        a_yes_b_no = int(
            ((disagreements[col_a] == "YES") & (disagreements[col_b] == "NO")).sum()
        )
        a_no_b_yes = int(
            ((disagreements[col_a] == "NO") & (disagreements[col_b] == "YES")).sum()
        )
        print(f"\n  {model_a}=YES / {model_b}=NO: {a_yes_b_no} chunks")
        print(f"  {model_a}=NO  / {model_b}=YES: {a_no_b_yes} chunks")

        print("\nExample disagreements (first 3):")
        print("-" * 60)
        for i, (_, row) in enumerate(disagreements.head(3).iterrows(), 1):
            preview = (
                row["text"][:200] + "..." if len(row["text"]) > 200 else row["text"]
            )
            print(f"\nExample #{i} — Chunk ID: {row['chunk_id']}")
            print(f"  {model_a}: {row[col_a]}  |  {model_b}: {row[col_b]}")
            print(f"  {preview}")

    print("\nTheme files: outputs/07_themes_model_a.txt / 07_themes_model_b.txt")


if __name__ == "__main__":
    main()
