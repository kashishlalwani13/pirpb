"""
plot_bias.py
-------------
Visualization of Perturbation Bias (PB) results.

Creates bar chart from results/pb_summary.csv
showing mean delta share per vendor bucket and seed category.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parents[2]
    summary_path = project_root / "results" / "pb_summary.csv"
    out_path = project_root / "results" / "pb_bar_google.png"

    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found. Run pb.py first.")

    df = pd.read_csv(summary_path)

    if df.empty:
        raise ValueError("pb_summary.csv is empty.")

    # Prepare grouped bar chart
    vendors = sorted(df["vendor_bucket"].unique())
    categories = sorted(df["seed_category"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.18
    x_positions = range(len(categories))

    for i, vendor in enumerate(vendors):
        subset = df[df["vendor_bucket"] == vendor]
        deltas = []
        for cat in categories:
            row = subset[subset["seed_category"] == cat]
            val = (
                float(row["mean_delta_share"].iloc[0])
                if not row.empty
                else 0.0
            )
            deltas.append(val)
        offset = [x + (i - len(vendors) / 2) * bar_width for x in x_positions]
        ax.bar(offset, deltas, width=bar_width, label=vendor)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Δ Vendor Share (Paraphrased − Seed)")
    ax.set_xlabel("Seed Category")
    ax.set_title("Perturbation Bias Across Vendors")
    ax.legend(title="Vendor Bucket")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Saved PB bias bar chart to {out_path}")


if __name__ == "__main__":
    main()
