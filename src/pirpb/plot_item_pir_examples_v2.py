# src/pirpb/plot_item_pir_examples_v2.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")


def plot_top_examples(n: int = 8):
    # Uses item_pir_google.csv produced by item_pir_v2.py
    df = pd.read_csv(RESULTS_DIR / "item_pir_google.csv")

    required_cols = {
        "seed_query",
        "baseline_query",
        "paraphrase_query",
        "category",
        "similarity_bin",
        "baseline_rank",
        "para_rank",
        "impact_score",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in item_pir_google.csv: {missing}")

    # Only consider rows where there is some change (impact > 0)
    df = df[df["impact_score"] > 0].copy()
    if df.empty:
        print("[WARN] No positive-impact rows to plot.")
        return

    # Sort by highest impact
    df = df.sort_values("impact_score", ascending=False)

    top = df.head(n)

    for i, (_, row) in enumerate(top.iterrows(), start=1):
        seed = str(row["seed_query"])
        baseline_q = str(row["baseline_query"])
        para_q = str(row["paraphrase_query"])
        cat = str(row["category"]).strip().capitalize()
        sim_bin = str(row["similarity_bin"]).strip().lower()

        baseline_rank = int(row["baseline_rank"])
        para_rank = int(row["para_rank"])
        impact = float(row["impact_score"])

        labels = ["baseline", "paraphrase"]
        ranks = [baseline_rank, para_rank]

        plt.figure(figsize=(4, 4))
        plt.bar(labels, ranks)
        plt.gca().invert_yaxis()  # rank 1 at the top
        plt.ylabel("Rank (lower is better)")
        plt.title(
            f"{cat} | {seed[:35]}...\n"
            f"{sim_bin} similarity, impact={impact:.2f}"
        )
        plt.tight_layout()

        out_path = RESULTS_DIR / f"example_rank_shift_{i}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"[OK] Saved {out_path}")
        print(f"     Seed: {seed}")
        print(f"     Baseline: {baseline_q}")
        print(f"     Paraphrase: {para_q}")
        print(
            f"     Baseline rank={baseline_rank}, Paraphrase rank={para_rank}, Impact={impact:.2f}")


if __name__ == "__main__":
    plot_top_examples()
