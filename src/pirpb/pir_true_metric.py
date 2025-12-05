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
    """
    Compute 'true' PIR using Kevin's definition:

        Query Distance  = 1 - cosine_similarity
        Search Distance = |impact_score|
        True PIR        = Query Distance / Search Distance

    With the convention:
        - If Search Distance == 0 -> PIR = 1 (max stability)
        - If Search Distance > 0  -> PIR in (0, 1] given current setup

    Also computes Perturbation Bias (PB):

        mean_pir(category)   = average PIR over items in that category
        overall_mean_pir     = average PIR over all items
        PB(category)         = mean_pir(category) / overall_mean_pir
    """
    # 1. Load data
    item_df = pd.read_csv(item_pir_path)
    para_df = pd.read_csv(paraphrases_path)

    # ------------------------------------------------------------------
    # Normalize paraphrase dataframe schema so it matches expectations.
    #
    # Current paraphrases.csv has columns:
    #   ['category', 'query', 'paraphrase', 'cosine', 'similarity_bin']
    #
    # Older / v2 formats might already have:
    #   ['base_query', 'paraphrase_query', 'cosine_similarity', ...]
    # ------------------------------------------------------------------
    if "paraphrase_query" in para_df.columns and "cosine_similarity" in para_df.columns:
        # Already in the expected format (e.g., paraphrases_v2)
        pass
    elif "paraphrase" in para_df.columns and "cosine" in para_df.columns:
        # Rename current columns to the expected names
        para_df = para_df.rename(
            columns={"paraphrase": "paraphrase_query",
                     "cosine": "cosine_similarity"}
        )
    else:
        raise ValueError(
            "Unexpected columns in paraphrases file. "
            f"Got columns: {para_df.columns.tolist()}. "
            "Expected either ['paraphrase_query', 'cosine_similarity', ...] "
            "or ['paraphrase', 'cosine', ...]."
        )

    # ------------------------------------------------------------------
    # 2. Merge cosine_similarity into item-level dataframe
    #    We join on the paraphrased query text.
    #
    # item_df: ... paraphrase_query, category, similarity_bin, impact_score
    # para_df: ... paraphrase_query, cosine_similarity
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Query Distance = 1 - cosine_similarity
    # ------------------------------------------------------------------
    merged["query_distance"] = 1.0 - merged["cosine_similarity"]

    # ------------------------------------------------------------------
    # 4. Search Distance = |impact_score|
    # ------------------------------------------------------------------
    merged["search_distance"] = merged["impact_score"].abs()

    # ------------------------------------------------------------------
    # 5. True PIR = Query Distance / Search Distance
    #
    # Convention from Kevin:
    #   - If Search Distance == 0 -> PIR = 1 (max stability)
    #   - Else                    -> PIR = Query Distance / Search Distance
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Save per-item PIR
    # ------------------------------------------------------------------
    output_pir_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_pir_path, index=False)
    print(f"[INFO] Saved per-item true PIR to {output_pir_path}")

    # ------------------------------------------------------------------
    # 7. Compute Perturbation Bias (PB)
    #
    # PB(category) = mean_pir(category) / overall_mean_pir
    # ------------------------------------------------------------------
    if "category" not in merged.columns:
        raise ValueError(
            "Column 'category' is missing from item-level PIR dataframe. "
            "Cannot compute PB without categories."
        )

    # Mean PIR by category
    cat_df = (
        merged.groupby("category")["pir_true"]
        .mean()
        .reset_index()
        .rename(columns={"pir_true": "mean_pir_true"})
    )

    if len(cat_df) == 0:
        overall_mean_pir_true = np.nan
    else:
        overall_mean_pir_true = merged["pir_true"].mean()

    cat_df["overall_mean_pir_true"] = overall_mean_pir_true
    cat_df["pb"] = cat_df["mean_pir_true"] / overall_mean_pir_true

    output_pb_path.parent.mkdir(parents=True, exist_ok=True)
    cat_df.to_csv(output_pb_path, index=False)
    print(f"[INFO] Saved PB summary to {output_pb_path}")

    print("[DEBUG] Category summary:")
    print(cat_df)

    # ------------------------------------------------------------------
    # 8. PB bar chart
    # ------------------------------------------------------------------
    if not cat_df.empty and cat_df["pb"].notna().any():
        plt.figure()
        plt.bar(cat_df["category"], cat_df["pb"])
        plt.xlabel("Category")
        plt.ylabel("Perturbation Bias (PB)")
        plt.title("Perturbation Bias by Category")
        plt.tight_layout()
        output_pb_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_pb_plot)
        plt.close()
        print(f"[INFO] Saved PB bar chart to {output_pb_plot}")
    else:
        print("[INFO] PB bar chart not created because category PB data is empty.")

    # ------------------------------------------------------------------
    # 9. Query vs Search Distance scatter plot
    # ------------------------------------------------------------------
    if not merged.empty:
        plt.figure()
        plt.scatter(merged["query_distance"],
                    merged["search_distance"], alpha=0.6)
        plt.xlabel("Query Distance (1 - cosine_similarity)")
        plt.ylabel("Search Distance (|impact_score|)")
        plt.title("Query Distance vs Search Distance")
        plt.tight_layout()
        output_scatter_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_scatter_plot)
        plt.close()
        print(
            f"[INFO] Saved Query vs Search Distance scatter to {output_scatter_plot}")
    else:
        print("[INFO] Scatter plot not created because merged PIR data is empty.")


def main() -> None:
    # Assume script is run from project root (`/Users/.../pirpb`)
    item_pir_path = Path("results/item_pir_google.csv")
    paraphrases_path = Path("data/paraphrases/paraphrases.csv")

    output_pir_path = Path("results/pir_true_google.csv")
    output_pb_path = Path("results/pb_google.csv")
    output_pb_plot = Path("results/pb_bar_google.png")
    output_scatter_plot = Path("results/query_vs_search_google.png")

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
