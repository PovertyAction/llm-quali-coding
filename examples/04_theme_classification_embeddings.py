from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.coding import (
    add_theme_similarity_columns,
    classify_by_max_theme,
    embed_themes,
    load_themes,
)
from src.openai_client import get_client


def generate_html_report(df: pd.DataFrame, themes: list, output_path: Path) -> None:
    """Generate an interactive HTML report of theme classification."""
    theme_cols = [t.short_name for t in themes]
    theme_full_names = {t.short_name: t.full_definition for t in themes}

    total_chunks = len(df)
    total_themes = len(theme_cols)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clasificación Temática - Resultados</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            cursor: pointer;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
            user-select: none;
        }}
        h2:hover {{
            background: #dfe6e9;
        }}
        .theme-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .theme-card {{
            background: #fff;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .theme-card h3 {{
            margin: 0 0 10px 0;
            color: #3498db;
            font-size: 0.9em;
        }}
        .theme-card .count {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .theme-card .avg-score {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .chunk {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .chunk-meta {{
            font-size: 0.85em;
            color: #7f8c8d;
            margin-bottom: 8px;
        }}
        .score {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.85em;
        }}
        .chunk-text {{
            color: #2c3e50;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }}
        .theme-content {{
            display: none;
            margin-top: 10px;
        }}
        .theme-content.active {{
            display: block;
        }}
        .toggle-indicator {{
            float: right;
            font-size: 0.8em;
            color: #7f8c8d;
        }}
        .stats {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Clasificación Temática de Chunks</h1>

        <div class="stats">
            <strong>Total de chunks analizados:</strong> {total_chunks}<br>
            <strong>Total de temas:</strong> {total_themes}
        </div>

        <h2>Resumen por Tema</h2>
        <div class="theme-summary">
"""

    # Add theme summary cards
    for theme_col in theme_cols:
        theme_df = df[df["most_similar_theme"] == theme_col]
        count = len(theme_df)
        avg_score = theme_df[theme_col].mean() if count > 0 else 0

        html += f"""
            <div class="theme-card">
                <h3>{theme_col}</h3>
                <div class="count">{count}</div>
                <div class="avg-score">Score promedio: {avg_score:.3f}</div>
            </div>
"""

    html += """
        </div>

        <h2>Chunks por Tema</h2>
"""

    # Add detailed sections for each theme
    for idx, theme_col in enumerate(theme_cols):
        theme_df = df[df["most_similar_theme"] == theme_col].sort_values(
            by=theme_col, ascending=False
        )

        if len(theme_df) == 0:
            continue

        full_name = theme_full_names.get(theme_col, theme_col)

        html += f"""
        <h2 onclick="toggleTheme('theme-{idx}')">
            {theme_col} ({len(theme_df)} chunks)
            <span class="toggle-indicator">▼ Click para expandir</span>
        </h2>
        <div id="theme-{idx}" class="theme-content">
            <p><strong>Definición:</strong> {full_name}</p>
"""

        # Show top 5 chunks for this theme
        for _, row in theme_df.head(10).iterrows():
            chunk_text = (
                row["text"][:500] + "..." if len(row["text"]) > 500 else row["text"]
            )
            html += f"""
            <div class="chunk">
                <div class="chunk-meta">
                    <span class="score">Score: {row[theme_col]:.3f}</span>
                    Chunk ID: {row["chunk_id"]}
                </div>
                <div class="chunk-text">{chunk_text}</div>
            </div>
"""

        html += """
        </div>
"""

    html += """
    </div>

    <script>
        function toggleTheme(id) {
            const content = document.getElementById(id);
            content.classList.toggle('active');
        }
    </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    """Classify chunks by theme similarity using embeddings."""
    client = get_client()

    inp = Path("outputs/01_chunks_with_embeddings.csv")
    if not inp.exists():
        raise FileNotFoundError(
            "Missing outputs/01_chunks_with_embeddings.csv. Run step 02 first."
        )

    df = pd.read_csv(inp)
    df["embedding"] = df["embedding"].apply(json.loads)

    themes_path = Path("data/themes/help_themes.json")
    themes = load_themes(themes_path)
    themes = embed_themes(client, themes)

    df = add_theme_similarity_columns(df, themes)
    theme_cols = [t.short_name for t in themes]
    df = classify_by_max_theme(df, theme_cols, out_col="most_similar_theme")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "03_theme_classification.csv"

    df_to_save = df.copy()
    df_to_save["embedding"] = df_to_save["embedding"].apply(json.dumps)
    df_to_save.to_csv(out_path, index=False)

    print(f"✅ Wrote: {out_path}")

    # Generate HTML report
    html_path = out_dir / "03_theme_classification_report.html"
    generate_html_report(df, themes, html_path)
    print(f"✅ Wrote interactive report: {html_path}")

    print("\n" + "=" * 60)
    print("RESUMEN DE CLASIFICACIÓN TEMÁTICA")
    print("=" * 60)

    print(f"\nTotal de chunks analizados: {len(df)}")
    print(f"Total de temas: {len(theme_cols)}")

    print("\n📊 Distribución de chunks por tema:")
    print("-" * 60)
    counts = df["most_similar_theme"].value_counts()
    for theme, count in counts.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print(f"{theme:40} {count:4d} ({pct:5.1f}%) {bar}")

    # Print top examples per theme with better formatting
    print("\n" + "=" * 60)
    print("EJEMPLOS TOP POR TEMA (mejores 3 de cada uno)")
    print("=" * 60)

    for t in theme_cols:
        top = (
            df[df["most_similar_theme"] == t].sort_values(by=t, ascending=False).head(3)
        )
        if len(top) == 0:
            continue

        # Get theme full name
        theme_obj = next((th for th in themes if th.short_name == t), None)
        full_name = theme_obj.full_definition if theme_obj else t

        print(f"\n{'=' * 60}")
        print(f"🏷️  TEMA: {t}")
        print(f"📝 Definición: {full_name}")
        print(f"📊 Total de chunks: {len(df[df['most_similar_theme'] == t])}")
        print(f"{'=' * 60}")

        for i, (_, row) in enumerate(top.iterrows(), 1):
            chunk_preview = (
                row["text"][:300] + "..." if len(row["text"]) > 300 else row["text"]
            )
            print(
                f"\n   Ejemplo #{i} - Score: {row[t]:.3f} | Chunk ID: {row['chunk_id']}"
            )
            print(f"   {'-' * 56}")
            print(f"   {chunk_preview}")
            print()


if __name__ == "__main__":
    main()
