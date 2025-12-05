"""
stat_tests.py
--------------
Stage H – Comparative Evaluation & Statistical Analysis
Performs significance testing across vendor categories.

For now, this script is robust to the case where we only have
a single vendor category (e.g., only Google). In that case,
we skip ANOVA / Kruskal–Wallis and write a summary noting that
at least two groups are required.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def main() -> None:
    # Project root: .../pirpb
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"
    item_pir_path = results_dir / "item_pir_google.csv"
    out_csv = results_dir / "stat_tests_summary.csv"
    out_plot = results_dir / "stat_tests_plot.png"

    if not item_pir_path.exists():
        raise FileNotFoundError(f"Missing {item_pir_path}")

    df = pd.read_csv(item_pir_path)

    if "category" not in df.columns or "impact_score" not in df.columns:
        raise ValueError(
            "Expected columns 'category' and 'impact_score' in "
            f"{item_pir_path}, got {df.columns.tolist()}"
        )

    # Normalize category labels
    df["category"] = df["category"].str.lower()

    # --------------------------------------------------------------
    # Boxplot: Impact Score Distribution by Vendor Category
    # --------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    df.boxplot(column="impact_score", by="category", grid=False)
    plt.title("Impact Score Distribution by Vendor Category")
    plt.suptitle("")
    plt.ylabel("Impact Score (0–1)")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()

    # --------------------------------------------------------------
    # ANOVA & Kruskal–Wallis tests across categories
    # --------------------------------------------------------------
    categories = sorted(df["category"].unique())
    groups = [df[df["category"] == c]["impact_score"].dropna()
              for c in categories]

    if len(categories) < 2:
        # Not enough groups for between-category tests
        summary = pd.DataFrame(
            {
                "Test": ["ANOVA", "Kruskal–Wallis"],
                "Statistic": [float("nan"), float("nan")],
                "p_value": [float("nan"), float("nan")],
                "note": [
                    "Skipped: need ≥ 2 categories",
                    "Skipped: need ≥ 2 categories",
                ],
            }
        )
        print(
            "[INFO] Only one vendor category present "
            f"({categories[0]}). Skipping ANOVA/Kruskal–Wallis."
        )
    else:
        # Standard case: at least two vendor categories
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)

        summary = pd.DataFrame(
            {
                "Test": ["ANOVA", "Kruskal–Wallis"],
                "Statistic": [round(f_stat, 4), round(h_stat, 4)],
                "p_value": [round(p_anova, 6), round(p_kw, 6)],
                "note": ["OK", "OK"],
            }
        )

    summary.to_csv(out_csv, index=False)

    print(f"[OK] Statistical tests summary → {out_csv}")
    print(f"[OK] Boxplot saved → {out_plot}")


if __name__ == "__main__":
    main()
