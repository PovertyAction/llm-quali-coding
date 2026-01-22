from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def main() -> None:
    """Perform inductive clustering on relevant chunks."""
    inp = Path("outputs/04_relevant_chunks.csv")
    if not inp.exists():
        raise FileNotFoundError(
            "Missing outputs/04_relevant_chunks.csv. Run step 04 first."
        )

    df = pd.read_csv(inp)
    embeddings = np.vstack(df["embedding"].apply(json.loads).apply(np.array).values)

    n_clusters = min(4, len(df))
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    labels = kmeans.fit_predict(embeddings)
    df["cluster"] = labels

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "09_clusters.csv"
    df.to_csv(out_csv, index=False)

    # 2D visualization
    if len(df) >= 3:
        tsne = TSNE(
            n_components=2,
            perplexity=min(15, max(2, len(df) - 1)),
            random_state=42,
            init="random",
            learning_rate=200,
        )
        coords = tsne.fit_transform(embeddings)
        df["x"] = coords[:, 0]
        df["y"] = coords[:, 1]

        for c in sorted(df["cluster"].unique()):
            subset = df[df["cluster"] == c]
            plt.scatter(subset["x"], subset["y"], alpha=0.7, label=f"cluster {c}")

        plt.title("t-SNE visualization of chunk clusters")
        plt.legend()
        out_png = out_dir / "09_clusters_tsne.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote: {out_png}")

    print(f"Wrote: {out_csv}")
    print("\nCluster sizes:")
    print(df["cluster"].value_counts())


if __name__ == "__main__":
    main()
