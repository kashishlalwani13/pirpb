"""
pb.py
------
Perturbation Bias (PB) analysis.

Goal:
  For each seed query and its paraphrases, measure how the share of
  different vendor domains (Google, Microsoft, IBM, Neutral, Other)
  changes between:
    - the original seed query SERP
    - the paraphrased query SERPs

Inputs:
  - configs/domain_map.yaml
  - data/queries/base_queries.csv
  - data/paraphrases/paraphrases.csv
  - serp/serp_google.csv  (must contain both seeds + paraphrases)

Outputs:
  - results/pb_detailed.csv
      One row per (seed_query, seed_category, vendor_bucket)
      Columns: seed_share, para_mean_share, delta_share, etc.

  - results/pb_summary.csv
      Aggregated by seed_category and vendor_bucket.

Usage (from repo root):
  export PYTHONPATH=src
  python -m pirpb.pb
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .domain_utils import load_domain_map, DEFAULT_BUCKET


def load_top_k(project_root: Path) -> int:
    """
    Try to read top_k from configs/config.yaml:
      - search.top_k
      - metrics.top_k
    Fallback: 10
    """
    cfg_path = project_root / "configs" / "config.yaml"
    default_k = 10
    if not cfg_path.exists():
        return default_k

    try:
        import yaml  # type: ignore

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return default_k

        search_cfg = cfg.get("search", {}) or {}
        metrics_cfg = cfg.get("metrics", {}) or {}

        if "top_k" in search_cfg:
            return int(search_cfg["top_k"])
        if "top_k" in metrics_cfg:
            return int(metrics_cfg["top_k"])
    except Exception:
        return default_k

    return default_k


def add_vendor_bucket_column(serp_df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """
    Given serp_google.csv as a DataFrame with 'url' column,
    add a 'vendor_bucket' column using the domain map.
    """
    dm = load_domain_map(project_root)

    if "url" not in serp_df.columns:
        raise ValueError("serp_google.csv must contain a 'url' column.")

    def _map(url: str) -> str:
        return dm.get_bucket_for_url(str(url))

    serp_df = serp_df.copy()
    serp_df["vendor_bucket"] = serp_df["url"].map(_map)
    return serp_df


def compute_share(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Convert vendor_bucket -> count into vendor_bucket -> share (0..1).
    If there are no counts (no valid URLs), everyone gets 0.0.
    """
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in counts.keys()}
    return {k: counts[k] / float(total) for k in counts.keys()}


def pb_analysis(project_root: Path):
    """
    Core PB computation.

    For each seed query:
      - Look at its top-K results -> compute vendor shares.
      - For each paraphrase of that seed:
            compute vendor shares of its top-K results.
      - Average paraphrase shares per vendor.
      - PB per vendor = para_mean_share - seed_share.

    Then:
      - Save detailed rows to results/pb_detailed.csv
      - Save aggregated summary to results/pb_summary.csv
    """
    # Paths
    base_queries_path = project_root / "data" / "queries" / "base_queries.csv"
    paraphrases_path = project_root / "data" / "paraphrases" / "paraphrases.csv"
    serp_path = project_root / "serp" / "serp_google.csv"
    results_dir = project_root / "results"

    # Sanity checks
    for p in [base_queries_path, paraphrases_path, serp_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # Load data
    seeds_df = pd.read_csv(base_queries_path)
    paras_df = pd.read_csv(paraphrases_path)
    serp_df = pd.read_csv(serp_path)

    # Normalize column names
    seeds_df.columns = [c.strip().lower() for c in seeds_df.columns]
    paras_df.columns = [c.strip().lower() for c in paras_df.columns]
    serp_df.columns = [c.strip().lower() for c in serp_df.columns]

    if "query" not in seeds_df.columns or "category" not in seeds_df.columns:
        raise ValueError("base_queries.csv must have 'category' and 'query' columns.")
    if "query" not in paras_df.columns or "paraphrase" not in paras_df.columns:
        raise ValueError("paraphrases.csv must have 'query' and 'paraphrase' columns.")
    required_serp_cols = {"query", "rank", "url"}
    if not required_serp_cols.issubset(serp_df.columns):
        raise ValueError(
            f"serp_google.csv must contain columns {required_serp_cols}, found {serp_df.columns.tolist()}"
        )

    # Ensure rank is numeric
    serp_df["rank"] = pd.to_numeric(serp_df["rank"], errors="coerce")
    serp_df = serp_df.dropna(subset=["rank"]).copy()
    serp_df["rank"] = serp_df["rank"].astype(int)

    # Add vendor buckets
    serp_df = add_vendor_bucket_column(serp_df, project_root)

    # Determine top_k
    top_k = load_top_k(project_root)
    print(f"[INFO] PB analysis using top_k = {top_k}")

    # All possible vendor buckets we might see
    all_buckets: List[str] = sorted(
        set(serp_df["vendor_bucket"].unique().tolist()) | {DEFAULT_BUCKET}
    )

    detailed_rows = []

    # Loop over each seed query
    for _, seed in seeds_df.iterrows():
        seed_query = str(seed["query"]).strip()
        seed_cat = str(seed["category"]).strip().lower()

        # Seed SERP
        seed_serp = (
            serp_df[serp_df["query"] == seed_query]
            .sort_values("rank")
            .head(top_k)
        )

        if seed_serp.empty:
            print(
                f"[WARN] No SERP rows found for seed query: {seed_query!r}. Skipping.",
                file=sys.stderr,
            )
            continue

        # Count vendors in seed SERP
        seed_counts = {b: 0 for b in all_buckets}
        for b in seed_serp["vendor_bucket"]:
            seed_counts[b] = seed_counts.get(b, 0) + 1
        seed_shares = compute_share(seed_counts)

        # Paraphrases for this seed
        these_paras = paras_df[paras_df["query"] == seed_query]
        if these_paras.empty:
            print(
                f"[WARN] No paraphrases found for seed query: {seed_query!r}. Skipping PB for this seed.",
                file=sys.stderr,
            )
            continue

        # For each paraphrase, compute vendor shares
        per_para_shares: List[Dict[str, float]] = []
        para_with_results = 0

        for _, prow in these_paras.iterrows():
            p_query = str(prow["paraphrase"]).strip()
            if not p_query:
                continue

            p_serp = (
                serp_df[serp_df["query"] == p_query]
                .sort_values("rank")
                .head(top_k)
            )

            if p_serp.empty:
                continue

            para_with_results += 1
            counts = {b: 0 for b in all_buckets}
            for b in p_serp["vendor_bucket"]:
                counts[b] = counts.get(b, 0) + 1
            shares = compute_share(counts)
            per_para_shares.append(shares)

        if para_with_results == 0 or not per_para_shares:
            print(
                f"[WARN] No paraphrase SERP rows for seed query: {seed_query!r}. Skipping PB for this seed.",
                file=sys.stderr,
            )
            continue

        # Average paraphrase shares per vendor bucket
        avg_para_shares: Dict[str, float] = {}
        for b in all_buckets:
            vals = [s[b] for s in per_para_shares]
            avg_para_shares[b] = sum(vals) / float(len(vals)) if vals else 0.0

        # For each vendor bucket, record detailed PB row
        for b in all_buckets:
            seed_share = float(seed_shares.get(b, 0.0))
            para_share = float(avg_para_shares.get(b, 0.0))
            delta = para_share - seed_share

            detailed_rows.append(
                {
                    "seed_query": seed_query,
                    "seed_category": seed_cat,
                    "vendor_bucket": b,
                    "seed_share": round(seed_share, 4),
                    "para_mean_share": round(para_share, 4),
                    "delta_share": round(delta, 4),
                    "num_paraphrases": int(len(these_paras)),
                    "num_paraphrases_with_results": int(para_with_results),
                }
            )

    if not detailed_rows:
        raise ValueError(
            "No PB rows were generated. "
            "Check that serp_google.csv contains both seed and paraphrase queries, "
            "and that domain_map.yaml is configured correctly."
        )

    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_df = pd.DataFrame(detailed_rows)

    # Save detailed
    detailed_path = results_dir / "pb_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # Build summary: mean delta per (seed_category, vendor_bucket)
    summary_df = (
        detailed_df.groupby(["seed_category", "vendor_bucket"], as_index=False)[
            ["seed_share", "para_mean_share", "delta_share"]
        ]
        .mean()
        .rename(
            columns={
                "seed_share": "mean_seed_share",
                "para_mean_share": "mean_para_mean_share",
                "delta_share": "mean_delta_share",
            }
        )
    )

    summary_path = results_dir / "pb_summary.csv"
    summary_df.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"[OK] Wrote PB detailed results to: {detailed_path}")
    print(f"[OK] Wrote PB summary results to: {summary_path}")

    return detailed_df, summary_df


def main() -> None:
    # src/pirpb/pb.py -> parents[2] is repo root
    project_root = Path(__file__).resolve().parents[2]
    pb_analysis(project_root)


if __name__ == "__main__":
    main()
