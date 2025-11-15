from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_true_pir(
    item_pir_path: Path,
    paraphrases_path: Path,
    output_pir_path: Path,
    output_pb_path: Path,
    output_pb_plot: Path,
    output_scatter_plot: Path,
) -> None:
    # 1. Load data
    item_df = pd.read_csv(item_pir_path)
    para_df = pd.read_csv(paraphrases_path)

    # Expect:
    # item_df: ... paraphrase_query, category, similarity_bin, impact_score
    # para_df: ... paraphrase_query, cosine_similarity, similarity_bin, template_level

    # 2. Merge cosine_similarity into item-level dataframe
    merged = item_df.merge(
        para_df[["paraphrase_query", "cosine_similarity"]],
        on="paraphrase_query",
        how="left",
    )

    missing = merged["cosine_similarity"].isna().sum()
    print(f"[INFO] Total rows: {len(merged)}")
    print(f"[INFO] Rows with missing cosine_similarity after merge: {missing}")

    if missing > 0:
        print("[WARN] Some rows are missing cosine_similarity and will be dropped.")

    # Drop rows without cosine
    merged = merged.dropna(subset=["cosine_similarity"])

    # 3. Query Distance = 1 - cosine_similarity
    merged["query_distance"] = 1.0 - merged["cosine_similarity"]

    # 4. Search Distance = |impact_score|
    merged["search_distance"] = merged["impact_score"].abs()

    # 5. True PIR = Query Distance / Search Distance
    zero_mask = merged["search_distance"] == 0
    nonzero_mask = ~zero_mask

    merged["pir_true"] = np.nan
    merged.loc[nonzero_mask, "pir_true"] = (
        merged.loc[nonzero_mask, "query_distance"]
        / merged.loc[nonzero_mask, "search_distance"]
    )

    # Replace infinities and handle zeros
    merged["pir_true"] = merged["pir_true"].replace([np.inf, -np.inf], np.nan)
    merged.loc[zero_mask, "pir_true"] = 1.0

    # Drop any remaining NaNs in pir_true
    merged = merged.dropna(subset=["pir_true"])

    # 6. Save per-item PIR
    output_pir_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_pir_path, index=False)
    print(f"[INFO] Saved per-item true PIR to {output_pir_path}")

    # 7. Compute category-wise mean PIR and PB
    category_summary = (
        merged.groupby("category")["pir_true"]
        .mean()
        .reset_index(name="mean_pir_true")
    )

    overall_mean = merged["pir_true"].mean()
    category_summary["overall_mean_pir_true"] = overall_mean
    category_summary["pb"] = category_summary["mean_pir_true"] / overall_mean

    category_summary.to_csv(output_pb_path, index=False)
    print(f"[INFO] Saved PB summary to {output_pb_path}")
    print("[DEBUG] Category summary:")
    print(category_summary)

    # 8. PB bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(category_summary["category"], category_summary["pb"])
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Category (Vendor)")
    plt.ylabel("Perturbation Bias (PB)")
    plt.title("Perturbation Bias by Category (True PIR)")
    plt.tight_layout()
    output_pb_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pb_plot, dpi=200)
    plt.close()
    print(f"[INFO] Saved PB bar chart to {output_pb_plot}")

    # 9. Scatter: Query Distance vs Search Distance
    plt.figure(figsize=(6, 4))
    plt.scatter(merged["query_distance"], merged["search_distance"], alpha=0.4)
    plt.xlabel("Query Distance (1 - cosine)")
    plt.ylabel("Search Distance (|impact_score|)")
    plt.title("Query vs Search Distance (Google CSE)")
    plt.tight_layout()
    plt.savefig(output_scatter_plot, dpi=200)
    plt.close()
    print(
        f"[INFO] Saved Query vs Search Distance scatter to {output_scatter_plot}")


def main():
    project_root = Path(".")

    item_pir_path = project_root / "results" / "item_pir_google.csv"
    paraphrases_path = project_root / "data" / "paraphrases" / "paraphrases_v2.csv"

    output_pir_path = project_root / "results" / "pir_true_google.csv"
    output_pb_path = project_root / "results" / "pb_google.csv"
    output_pb_plot = project_root / "results" / "pb_bar_google.png"
    output_scatter_plot = project_root / "results" / "query_vs_search_google.png"

    compute_true_pir(
        item_pir_path=item_pir_path,
        paraphrases_path=paraphrases_path,
        output_pir_path=output_pir_path,
        output_pb_path=output_pb_path,
        output_pb_plot=output_pb_plot,
        output_scatter_plot=output_scatter_plot,
    )


if __name__ == "__main__":
    main()
