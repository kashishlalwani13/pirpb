"""
pir_by_bin.py
-------------
Stage D-ext: Compute RBO-based PIR separately for different
paraphrase similarity levels (high / medium / low), using an
anchor-based approach that works with paraphrase SERPs only.

For each seed query:
  - Take all its paraphrases from data/paraphrases/paraphrases.csv.
  - Choose the paraphrase with the highest cosine similarity
    as the "anchor" (baseline).
  - Use SERP(anchor) as reference.
  - For every other paraphrase of that seed:
        compute RBO(SERP(anchor), SERP(paraphrase))
        PIR = 1 - RBO
        tag with that paraphrase's similarity_bin.

We then aggregate mean PIR per similarity_bin, so we can see how
robustness changes for high vs medium vs low similarity paraphrases.

Inputs:
  - data/paraphrases/paraphrases.csv  (with columns:
        category, query, paraphrase, cosine, similarity_bin)
  - serp/serp_google.csv             (SERPs for paraphrases)
  - configs/config.yaml              (for io + rbo_p + top_k)

Outputs:
  - results/pir_by_bin_google.csv      detailed rows
  - results/pir_by_bin_summary.csv     mean PIR per bin
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from .metrics import rbo, pir_from_rbo


def _norm(s: str) -> str:
    return " ".join(str(s).strip().split()).lower() if isinstance(s, str) else ""


def _load_cfg(project_root: Path) -> dict:
    cfg_path = project_root / "configs" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    project_root = Path(__file__).resolve().parents[2]
    cfg = _load_cfg(project_root)

    serp_dir = project_root / cfg.get("io", {}).get("serp_dir", "serp")
    results_dir = project_root / cfg.get("io", {}).get("results_dir", "results")

    serp_path = serp_dir / "serp_google.csv"
    para_path = project_root / "data" / "paraphrases" / "paraphrases.csv"

    top_k = int(cfg.get("search", {}).get("top_k", 10))
    p = float(cfg.get("metrics", {}).get("rbo_p", 0.9))

    if not serp_path.exists():
        raise FileNotFoundError(f"Missing serp file: {serp_path}")
    if not para_path.exists():
        raise FileNotFoundError(f"Missing paraphrases file: {para_path}")

    # Load data
    serp = pd.read_csv(serp_path)
    para = pd.read_csv(para_path)

    # Normalize columns
    serp.columns = [c.strip().lower() for c in serp.columns]
    para.columns = [c.strip().lower() for c in para.columns]

    # Basic checks
    required_serp = {"query", "rank", "url"}
    if not required_serp.issubset(serp.columns):
        raise ValueError(
            f"serp_google.csv must contain {required_serp}, found {serp.columns.tolist()}"
        )

    for col in ["query", "paraphrase", "cosine", "similarity_bin"]:
        if col not in para.columns:
            raise ValueError(
                f"paraphrases.csv must contain '{col}' column. "
                "Regenerate with updated paraphraser if needed."
            )

    # Clean SERP
    serp = serp.dropna(subset=["query", "rank", "url"]).copy()
    serp["rank"] = pd.to_numeric(serp["rank"], errors="coerce")
    serp = serp.dropna(subset=["rank"]).copy()
    serp["rank"] = serp["rank"].astype(int)

    # Canonical query for SERP lookup
    serp["query_norm"] = serp["query"].map(_norm)

    # Map from canonical query -> topK URL list
    urls_by_q: Dict[str, List[str]] = {}
    for qn, grp in serp.sort_values("rank").groupby("query_norm"):
        urls_by_q[qn] = grp.head(top_k)["url"].astype(str).tolist()

    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Group paraphrases by original seed query
    for seed_query, grp in para.groupby("query"):
        grp = grp.copy()

        # Choose anchor = paraphrase with highest cosine for this seed
        anchor_row = grp.loc[grp["cosine"].idxmax()]
        anchor_text = str(anchor_row["paraphrase"]).strip()
        anchor_norm = _norm(anchor_text)
        anchor_urls = urls_by_q.get(anchor_norm, [])

        if not anchor_urls:
            # No SERP for anchor -> skip this seed
            continue

        seed_category = str(anchor_row.get("category", "")).strip().lower()

        # Compare all other paraphrases for this seed against anchor
        for _, prow in grp.iterrows():
            para_text = str(prow["paraphrase"]).strip()
            if not para_text or para_text == anchor_text:
                continue

            sim_bin = str(prow.get("similarity_bin", "")).strip().lower()
            para_norm = _norm(para_text)
            para_urls = urls_by_q.get(para_norm, [])

            if not para_urls:
                # No SERP for this paraphrase -> skip
                continue

            score = rbo(anchor_urls, para_urls, p)
            pir = pir_from_rbo(score)

            rows.append(
                {
                    "seed_query": seed_query,
                    "anchor_paraphrase": anchor_text,
                    "paraphrase": para_text,
                    "category": seed_category,
                    "similarity_bin": sim_bin,
                    "k_used": min(
                        top_k,
                        max(len(anchor_urls), len(para_urls)),
                    ),
                    "p": p,
                    "rbo": score,
                    "pir": pir,
                }
            )

    if not rows:
        raise ValueError(
            "No PIR rows computed in anchor-based PIR-by-bin. "
            "Check that serp_google.csv contains SERPs for the paraphrases "
            "from paraphrases.csv."
        )

    df = pd.DataFrame(rows)
    out_detail = results_dir / "pir_by_bin_google.csv"
    df.to_csv(out_detail, index=False)

    # Summary: mean PIR per similarity_bin
    summary = (
        df.dropna(subset=["pir"])
        .groupby("similarity_bin", as_index=False)["pir"]
        .mean()
        .rename(columns={"pir": "mean_pir"})
    )
    out_summary = results_dir / "pir_by_bin_summary.csv"
    summary.to_csv(out_summary, index=False)

    print(f"[OK] PIR-by-bin (anchor-based) details -> {out_detail}")
    print(f"[OK] PIR-by-bin (anchor-based) summary -> {out_summary}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
