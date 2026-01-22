from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.llm_tasks import code_nonverbal_cues
from src.openai_client import get_client


def generate_nonverbal_html_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generate an interactive HTML report of non-verbal cue coding."""
    total_chunks = len(df)
    chunks_with_cues = len(df[df["any_nonverbal_cue"] == "YES"])

    # Count by cue type
    cue_type_counts = df[df["any_nonverbal_cue"] == "YES"]["cue_type"].value_counts()

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Códigos No Verbales - Resultados</title>
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
            border-bottom: 3px solid #e74c3c;
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
        .stats {{
            background: #fee;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #e74c3c;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #e74c3c;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .cue-type-section {{
            margin: 30px 0;
        }}
        .chunk {{
            background: #f8f9fa;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .chunk-meta {{
            font-size: 0.85em;
            color: #7f8c8d;
            margin-bottom: 8px;
        }}
        .cue-badge {{
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.85em;
            margin-left: 10px;
        }}
        .chunk-text {{
            color: #2c3e50;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
            background: white;
            padding: 10px;
            border-radius: 3px;
        }}
        .no-cues {{
            color: #95a5a6;
            font-style: italic;
            padding: 20px;
            text-align: center;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .toggle-indicator {{
            float: right;
            font-size: 0.8em;
            color: #7f8c8d;
        }}
        .cue-content {{
            display: none;
            margin-top: 10px;
        }}
        .cue-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Análisis de Códigos No Verbales</h1>

        <div class="stats">
            <strong>Resumen General</strong>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_chunks}</div>
                    <div class="stat-label">Total chunks analizados</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{chunks_with_cues}</div>
                    <div class="stat-label">Chunks con señales no verbales</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{(chunks_with_cues / total_chunks * 100 if total_chunks > 0 else 0):.1f}%</div>
                    <div class="stat-label">Porcentaje con señales</div>
                </div>
            </div>
        </div>
"""

    # Add sections by cue type
    if len(cue_type_counts) > 0:
        html += """
        <h2>Tipos de Señales No Verbales</h2>
"""
        for idx, (cue_type, count) in enumerate(cue_type_counts.items()):
            if not cue_type or cue_type.strip() == "":
                continue

            chunks_with_this_cue = df[df["cue_type"] == cue_type]

            html += f"""
        <h2 onclick="toggleCue('cue-{idx}')">
            {cue_type} ({count} chunks)
            <span class="toggle-indicator">▼ Click para expandir</span>
        </h2>
        <div id="cue-{idx}" class="cue-content">
"""

            for _, row in chunks_with_this_cue.iterrows():
                chunk_text = row["text"]
                html += f"""
            <div class="chunk">
                <div class="chunk-meta">
                    Chunk ID: {row["chunk_id"]}
                    <span class="cue-badge">{cue_type}</span>
                </div>
                <div class="chunk-text">{chunk_text}</div>
            </div>
"""

            html += """
        </div>
"""
    else:
        html += """
        <div class="no-cues">
            No se detectaron señales no verbales en los chunks analizados.
        </div>
"""

    html += """
    </div>

    <script>
        function toggleCue(id) {
            const content = document.getElementById(id);
            content.classList.toggle('active');
        }
    </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    """Code non-verbal cues from full transcript using structured LLM output."""
    client = get_client()

    # Load full chunks (not just relevant ones)
    inp = Path("outputs/01_chunks_with_embeddings.csv")
    if not inp.exists():
        raise FileNotFoundError(
            "Missing outputs/01_chunks_with_embeddings.csv. Run step 02 first."
        )

    print(f"Reading chunks from: {inp}")
    df = pd.read_csv(inp)
    df["embedding"] = df["embedding"].apply(json.loads)

    print(f"\nAnalyzing {len(df)} chunks for non-verbal cues...")
    print("This may take a few minutes...\n")

    cue_yes: list[str] = []
    cue_type: list[str] = []

    for i, row in df.iterrows():
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(df)} chunks...")
        res = code_nonverbal_cues(client, row["text"])
        cue_yes.append(res.get("any_cues", "NO"))
        cue_type.append(res.get("cue_type", ""))

    df["any_nonverbal_cue"] = cue_yes
    df["cue_type"] = cue_type

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "05_nonverbal_coding.csv"

    df_to_save = df.copy()
    df_to_save["embedding"] = df_to_save["embedding"].apply(json.dumps)
    df_to_save.to_csv(out_path, index=False)

    print(f"\n✅ Wrote: {out_path}")

    # Generate HTML report
    html_path = out_dir / "05_nonverbal_coding_report.html"
    generate_nonverbal_html_report(df, html_path)
    print(f"✅ Wrote interactive report: {html_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESUMEN DE CÓDIGOS NO VERBALES")
    print("=" * 60)

    chunks_with_cues = df[df["any_nonverbal_cue"] == "YES"]
    print(f"\nTotal de chunks analizados: {len(df)}")
    print(f"Chunks con señales no verbales: {len(chunks_with_cues)}")
    print(f"Porcentaje: {len(chunks_with_cues) / len(df) * 100:.1f}%")

    if len(chunks_with_cues) > 0:
        print("\n📊 Distribución por tipo de señal:")
        print("-" * 60)
        cue_counts = df["cue_type"].value_counts()
        for cue, count in cue_counts.items():
            if cue and cue.strip() != "":
                print(f"{cue:40} {count:4d}")

        print("\n🎭 Ejemplos de chunks con señales no verbales (primeros 3):")
        print("=" * 60)
        for i, (_, row) in enumerate(chunks_with_cues.head(3).iterrows(), 1):
            print(f"\nEjemplo #{i} - Tipo: {row['cue_type']}")
            print(f"Chunk ID: {row['chunk_id']}")
            print("-" * 60)
            preview = (
                row["text"][:200] + "..." if len(row["text"]) > 200 else row["text"]
            )
            print(preview)
    else:
        print("\n⚠️  No se detectaron señales no verbales en los chunks analizados.")


if __name__ == "__main__":
    main()
