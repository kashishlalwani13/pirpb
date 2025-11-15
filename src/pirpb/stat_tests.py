"""
stat_tests.py
--------------
Stage H – Comparative Evaluation & Statistical Analysis
Performs significance testing across vendor categories.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def main():
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"
    item_pir_path = results_dir / "item_pir_google.csv"
    out_csv = results_dir / "stat_tests_summary.csv"
    out_plot = results_dir / "stat_tests_plot.png"

    if not item_pir_path.exists():
        raise FileNotFoundError(f"Missing {item_pir_path}")

    df = pd.read_csv(item_pir_path)
    df["category"] = df["category"].str.lower()

    # Boxplot
    plt.figure(figsize=(6,4))
    df.boxplot(column="impact_score", by="category", grid=False)
    plt.title("Impact Score Distribution by Vendor Category")
    plt.suptitle("")
    plt.ylabel("Impact Score (0–1)")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()

    # ANOVA & Kruskal–Wallis tests
    categories = sorted(df["category"].unique())
    groups = [df[df["category"] == c]["impact_score"].dropna() for c in categories]

    f_stat, p_anova = stats.f_oneway(*groups)
    h_stat, p_kw = stats.kruskal(*groups)

    summary = pd.DataFrame({
        "Test": ["ANOVA", "Kruskal–Wallis"],
        "Statistic": [round(f_stat, 4), round(h_stat, 4)],
        "p_value": [round(p_anova, 6), round(p_kw, 6)],
    })
    summary.to_csv(out_csv, index=False)

    print(f"[OK] Statistical tests complete → {out_csv}")
    print(f"[OK] Boxplot saved → {out_plot}")


if __name__ == "__main__":
    main()
