# ==========================================================
# item_pir_v2.py
# ==========================================================
# Item-level, rank-based PIR (NO RBO) — Updated to use
# paraphrases_v2.csv (with high / medium / low bins).
# ==========================================================

from __future__ import annotations
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
from urllib.parse import urlparse
import yaml


# ----------------- helpers -----------------

def load_config_top_k(project_root: Path) -> int:
    cfg_path = project_root / "configs" / "config.yaml"
    default_k = 20
    try:
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                return default_k
            for key in ("item_pir", "metrics", "search"):
                if key in cfg and isinstance(cfg[key], dict) and "top_k" in cfg[key]:
                    return int(cfg[key]["top_k"])
    except Exception:
        pass
    return default_k


def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def find_rank_for_target(
    serp_df: pd.DataFrame,
    query_text: str,
    focal_type: str,
    focal_value: str,
    k: int,
) -> int:
    """Return rank (1..k) of focal item for given query_text, or k+1 if not found."""
    rows = serp_df[serp_df["query"] == query_text].copy()
    if rows.empty:
        return k + 1
    rows["rank"] = pd.to_numeric(rows["rank"], errors="coerce")
    rows = rows.dropna(subset=["rank"])
    if rows.empty:
        return k + 1
    rows["rank"] = rows["rank"].astype(int)
    rows = rows.sort_values("rank")

    fv = focal_value.strip().lower()
    for _, r in rows.iterrows():
        rank = int(r["rank"])
        if rank < 1 or rank > k:
            continue
        url = str(r.get("url", "")).strip().lower()
        if not url:
            continue
        if focal_type == "url":
            if fv in url:
                return rank
        elif focal_type == "domain":
            dom = extract_domain(url)
            if dom == fv or dom.endswith("." + fv):
                return rank
    return k + 1


# ----------------- main logic -----------------

def compute_item_pir(
    base_queries_path: Path,
    paraphrases_path: Path,
    focal_items_path: Path,
    serp_path: Path,
    results_dir: Path,
    k: int,
):
    # --- load ---
    if not base_queries_path.exists():
        raise FileNotFoundError(base_queries_path)
    if not paraphrases_path.exists():
        raise FileNotFoundError(paraphrases_path)
    if not focal_items_path.exists():
        raise FileNotFoundError(focal_items_path)
    if not serp_path.exists():
        raise FileNotFoundError(serp_path)

    seeds_df = pd.read_csv(base_queries_path)
    paras_df = pd.read_csv(paraphrases_path)
    focal_df = pd.read_csv(focal_items_path)
    serp_df = pd.read_csv(serp_path)

    # normalize columns
    for df in (seeds_df, paras_df, focal_df, serp_df):
        df.columns = [c.strip().lower() for c in df.columns]

    # checks
    required_focal = {"query", "category", "focal_type", "focal_value"}
    if not required_focal.issubset(focal_df.columns):
        missing = required_focal.difference(focal_df.columns)
        raise ValueError(
            f"focal_items.csv missing: {', '.join(sorted(missing))}")

    if "similarity_bin" not in paras_df.columns:
        raise ValueError(
            "paraphrases_v2.csv must contain 'similarity_bin'. Run paraphraser_v2 first."
        )

    serp_df = serp_df.dropna(subset=["query", "rank", "url"]).copy()
    serp_df["rank"] = pd.to_numeric(serp_df["rank"], errors="coerce")
    serp_df = serp_df.dropna(subset=["rank"])
    serp_df["rank"] = serp_df["rank"].astype(int)

    results_dir.mkdir(parents=True, exist_ok=True)

    # build focal lookup
    focal_lookup: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for _, r in focal_df.iterrows():
        q = str(r["query"]).strip()
        cat = str(r["category"]).strip().lower()
        ft = str(r["focal_type"]).strip().lower()
        fv = str(r["focal_value"]).strip()
        if q and cat and ft in {"url", "domain"} and fv:
            focal_lookup[(q, cat)] = (ft, fv)

    if not focal_lookup:
        raise ValueError("No valid focal mappings in focal_items.csv")

    item_rows: List[Dict] = []
    skipped_no_focal = 0

    # --- per focal seed ---
    for (seed_query, seed_cat_l), (f_type, f_val) in focal_lookup.items():
        # paraphrases for this seed
        pgrp = paras_df[paras_df["base_query"] == seed_query].copy()
        if pgrp.empty:
            print(
                f"[WARN] No paraphrases for seed: {seed_query!r}", file=sys.stderr)
            continue

        # 1) try true seed baseline
        seed_rank = find_rank_for_target(
            serp_df=serp_df,
            query_text=seed_query,
            focal_type=f_type,
            focal_value=f_val,
            k=k,
        )
        baseline_query = seed_query
        baseline_rank = seed_rank

        # 2) fallback to best paraphrase if seed missing
        if baseline_rank == k + 1:
            best_rank = k + 1
            best_q = None
            for _, prow in pgrp.iterrows():
                pq = str(prow["paraphrase_query"]).strip()
                if not pq:
                    continue
                rnk = find_rank_for_target(
                    serp_df=serp_df,
                    query_text=pq,
                    focal_type=f_type,
                    focal_value=f_val,
                    k=k,
                )
                if rnk < best_rank:
                    best_rank = rnk
                    best_q = pq
            if best_q is not None and best_rank <= k:
                baseline_query = best_q
                baseline_rank = best_rank

        # 3) if still no focal found anywhere, skip
        if baseline_rank == k + 1:
            skipped_no_focal += 1
            continue

        # 4) compute impact
        for _, prow in pgrp.iterrows():
            para_q = str(prow["paraphrase_query"]).strip()
            if not para_q:
                continue
            sim_bin = str(prow.get("similarity_bin", "")).strip().lower()
            para_rank = find_rank_for_target(
                serp_df=serp_df,
                query_text=para_q,
                focal_type=f_type,
                focal_value=f_val,
                k=k,
            )

            raw_delta = (para_rank - baseline_rank) / float(k)
            if raw_delta <= 0:
                impact = 0.0
            elif raw_delta >= 1:
                impact = 1.0
            else:
                impact = float(raw_delta)

            item_rows.append(
                {
                    "seed_query": seed_query,
                    "baseline_query": baseline_query,
                    "paraphrase_query": para_q,
                    "category": seed_cat_l,
                    "similarity_bin": sim_bin,
                    "focal_type": f_type,
                    "focal_value": f_val,
                    "baseline_rank": int(baseline_rank),
                    "para_rank": int(para_rank),
                    "delta_rank": int(para_rank - baseline_rank),
                    "impact_score": round(impact, 4),
                }
            )

    if not item_rows:
        raise ValueError(
            "No item-level PIR rows generated — missing focal appearances.")

    item_df = pd.DataFrame(item_rows)
    summary_df = (
        item_df.groupby(["category", "similarity_bin"],
                        as_index=False)["impact_score"]
        .mean()
        .rename(columns={"impact_score": "mean_impact_score"})
    )

    # save
    item_output = results_dir / "item_pir_google.csv"
    summary_output = results_dir / "item_pir_summary.csv"
    item_df.to_csv(item_output, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_df.to_csv(summary_output, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"[OK] Item-level rank PIR (v2) -> {item_output}")
    print(f"[OK] Item-level PIR summary (v2) -> {summary_output}")
    print(summary_df.to_string(index=False))
    print(f"[INFO] Seeds skipped due to no focal found: {skipped_no_focal}")

    return item_df, summary_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    base_queries_path = project_root / "data" / "queries" / "base_queries.csv"
    paraphrases_path = project_root / "data" / "paraphrases" / "paraphrases_v2.csv"
    focal_items_path = project_root / "data" / "queries" / "focal_items.csv"
    serp_path = project_root / "serp" / "serp_google.csv"
    results_dir = project_root / "results"

    k = load_config_top_k(project_root)
    print(f"[INFO] Using top_k = {k} for item-level rank PIR (v2).")

    compute_item_pir(
        base_queries_path=base_queries_path,
        paraphrases_path=paraphrases_path,
        focal_items_path=focal_items_path,
        serp_path=serp_path,
        results_dir=results_dir,
        k=k,
    )


if __name__ == "__main__":
    main()
