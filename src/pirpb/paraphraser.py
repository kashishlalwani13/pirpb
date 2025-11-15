# ==========================================================
# paraphraser.py  |  Stage B - Entity-safe paraphrasing
# ==========================================================
# Updated to address supervisor feedback:
# - Generate richer paraphrases (not just minor suffixes).
# - Use SentenceTransformer cosine similarity to label
#   perturbation strength into bins:
#       high   : cos >= 0.9
#       medium : 0.7 <= cos < 0.9
#       low    : cosine_min <= cos < 0.7
#
# Output CSV: data/paraphrases/paraphrases.csv
# Columns: category, query, paraphrase, cosine, similarity_bin
# ----------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import re
from typing import List, Dict

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from .config import load_settings


# ---------- Helpers ----------


def _protect_entities(text: str, entities: List[str]) -> tuple[str, List[tuple[str, str]]]:
    """
    Replace each locked entity with a placeholder ENT_i so templates
    do not disturb vendor/product names.
    """
    protected_map: List[tuple[str, str]] = []
    tmp = text
    for i, ent in enumerate(entities):
        ent = ent.strip()
        if not ent:
            continue
        placeholder = f"ENT_{i}"
        # case-insensitive exact match
        tmp = re.sub(re.escape(ent), placeholder, tmp, flags=re.IGNORECASE)
        protected_map.append((placeholder, ent))
    return tmp, protected_map


def _restore_entities(text: str, protected: List[tuple[str, str]]) -> str:
    out = text
    for placeholder, ent in protected:
        out = out.replace(placeholder, ent)
    return out


def _candidate_templates(core: str) -> List[str]:
    """
    Generate a richer set of paraphrase patterns around `core`.
    These patterns introduce:
      - how-to / tutorial forms
      - documentation / reference emphasis
      - example-driven intents
      - slight scope shifts while keeping overall intent
    """
    return [
        # close / high-similarity style
        f"{core} documentation",
        f"{core} official docs",
        f"{core} reference guide",
        f"{core} detailed guide",
        # medium changes
        f"how to get started with {core}",
        f"step-by-step tutorial for {core}",
        f"examples and best practices for {core}",
        f"{core} developer guide",
        f"introduction to {core} for beginners",
        # lower similarity but still relevant
        f"troubleshooting and FAQs for {core}",
        f"community tutorials and blog posts about {core}",
        f"use cases and sample projects using {core}",
        f"comparison of tools and services related to {core}",
    ]


def _similarity_bin(cos: float, cosine_min: float) -> str:
    if cos >= 0.9:
        return "high"
    if cos >= 0.7:
        return "medium"
    if cos >= cosine_min:
        return "low"
    return "discard"


# ---------- Main ----------


def main():
    settings = load_settings()
    in_csv = Path(settings.data_dir) / "queries" / "base_queries.csv"
    out_csv = Path(settings.data_dir) / "paraphrases" / "paraphrases.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    required_cols = {"category", "query", "lock_entities"}
    assert required_cols.issubset(df.columns), \
        f"CSV must contain columns: {required_cols}"

    model = SentenceTransformer(settings.embed_model)

    rows: List[Dict] = []

    for _, row in df.iterrows():
        category = str(row["category"]).strip()
        query = str(row["query"]).strip()
        locks = [x.strip()
                 for x in str(row["lock_entities"]).split("|") if x.strip()]

        # Protect entities
        protected_query, protected_map = _protect_entities(query, locks)

        # Generate candidate paraphrases around the protected core
        candidates = _candidate_templates(protected_query)

        # Restore entities back into each candidate
        candidates = [
            _restore_entities(c, protected_map)
            for c in candidates
        ]

        # Encode seed once
        base_emb = model.encode([query], normalize_embeddings=True)[0]

        # Score candidates and keep those above cosine_min
        for cand in candidates:
            if cand.strip().lower() == query.strip().lower():
                continue
            par_emb = model.encode([cand], normalize_embeddings=True)[0]
            cos = float(util.cos_sim(base_emb, par_emb))

            bin_label = _similarity_bin(cos, settings.cosine_min)
            if bin_label == "discard":
                continue

            rows.append(
                {
                    "category": category,
                    "query": query,
                    "paraphrase": cand,
                    "cosine": round(cos, 4),
                    "similarity_bin": bin_label,
                }
            )

    out_df = pd.DataFrame(
        rows,
        columns=["category", "query", "paraphrase",
                 "cosine", "similarity_bin"],
    )

    out_df.to_csv(out_csv, index=False)

    # Simple summary to stdout
    total = len(out_df)
    high = (out_df["similarity_bin"] == "high").sum()
    med = (out_df["similarity_bin"] == "medium").sum()
    low = (out_df["similarity_bin"] == "low").sum()

    print(
        f"Saved paraphrases to {out_csv} | total={total} "
        f"(high={high}, medium={med}, low={low}, cos_min={settings.cosine_min})"
    )


if __name__ == "__main__":
    main()
