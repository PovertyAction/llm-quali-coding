from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def main() -> None:
    """Perform inductive clustering on all chunks."""
    inp = Path("outputs/01_chunks_with_embeddings.csv")
    if not inp.exists():
        raise FileNotFoundError(
            "Missing outputs/01_chunks_with_embeddings.csv. Run step 02 first."
        )

    print(f"Reading chunks from: {inp}")
    df = pd.read_csv(inp)
    embeddings = np.vstack(df["embedding"].apply(json.loads).apply(np.array).values)

    n_clusters = min(8, len(df))
    print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    labels = kmeans.fit_predict(embeddings)
    df["cluster"] = labels

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "06_clusters.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n✅ Wrote: {out_csv}")

    # 2D visualization
    if len(df) >= 3:
        print("\nGenerating t-SNE visualization...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(2, len(df) - 1)),
            random_state=42,
            init="random",
            learning_rate=200,
        )
        coords = tsne.fit_transform(embeddings)
        df["x"] = coords[:, 0]
        df["y"] = coords[:, 1]

        plt.figure(figsize=(12, 8))
        for c in sorted(df["cluster"].unique()):
            subset = df[df["cluster"] == c]
            plt.scatter(subset["x"], subset["y"], alpha=0.6, s=50, label=f"Cluster {c}")

        plt.title(
            "t-SNE Visualization of Chunk Clusters", fontsize=14, fontweight="bold"
        )
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        out_png = out_dir / "06_clusters_tsne.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ Wrote: {out_png}")

    print("\n" + "=" * 60)
    print("RESUMEN DE CLUSTERING")
    print("=" * 60)
    print(f"\nTotal de chunks: {len(df)}")
    print(f"Número de clusters: {n_clusters}")

    print("\n📊 Tamaño de cada cluster:")
    print("-" * 60)
    cluster_counts = df["cluster"].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print(f"Cluster {cluster}:  {count:4d} chunks ({pct:5.1f}%) {bar}")

    print("\n📝 Ejemplos de chunks por cluster (primero de cada uno):")
    print("=" * 60)
    for c in sorted(df["cluster"].unique()):
        cluster_chunks = df[df["cluster"] == c]
        print(f"\n--- Cluster {c} ({len(cluster_chunks)} chunks) ---")
        first_chunk = cluster_chunks.iloc[0]
        preview = (
            first_chunk["text"][:300] + "..."
            if len(first_chunk["text"]) > 300
            else first_chunk["text"]
        )
        print(f"Chunk ID: {first_chunk['chunk_id']}")
        print(preview)
        print()


if __name__ == "__main__":
    main()
