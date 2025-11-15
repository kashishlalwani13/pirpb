# src/pirpb/plot_item_pir_summary_v2.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")


def main():
    df = pd.read_csv(RESULTS_DIR / "item_pir_google.csv")

    required = {"category", "similarity_bin", "impact_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in item_pir_google.csv: {missing}")

    # Normalize
    df["category"] = df["category"].astype(str).str.lower()
    df["similarity_bin"] = df["similarity_bin"].astype(str).str.lower()

    # Aggregate: mean + std + n
    summary = (
        df.groupby(["category", "similarity_bin"])["impact_score"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )

    print("[INFO] Item-level PIR summary (recomputed from rows):")
    print(summary.to_string(index=False))

    # Plot: grouped bar chart by vendor, colored by similarity_bin
    sim_bins = sorted(summary["similarity_bin"].unique())
    vendors = ["google", "microsoft", "ibm"]

    x = range(len(vendors))
    width = 0.18

    plt.figure(figsize=(8, 5))

    for i, sim in enumerate(sim_bins):
        sub = summary[summary["similarity_bin"] == sim]
        # align by vendor order; default to 0 if missing combo
        heights = []
        for v in vendors:
            row = sub[sub["category"] == v]
            heights.append(float(row["mean"]) if not row.empty else 0.0)

        offset = (i - (len(sim_bins) - 1) / 2) * width
        bar_x = [xi + offset for xi in x]
        plt.bar(bar_x, heights, width=width, label=f"{sim}")

    plt.xticks(list(x), [v.capitalize() for v in vendors])
    plt.ylim(0, 1.05)
    plt.ylabel("Mean impact_score (0 = stable, 1 = severe demotion)")
    plt.title("Item-level PIR by Vendor × Similarity Bin (v2 paraphrases)")
    plt.legend(title="similarity_bin")
    plt.tight_layout()

    out_path = RESULTS_DIR / "item_pir_summary_plot_v2.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
