# src/pirpb/paraphraser_v2.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import yaml

# =========================
# CONFIG & PATHS
# =========================
HIGH_MIN = 0.90
MED_MIN, MED_MAX = 0.70, 0.90
LOW_MIN, LOW_MAX = 0.50, 0.70

OUT_PATH = Path("data/paraphrases/paraphrases_v2.csv")
CONFIG_PATH = Path("configs/config.yaml")


# =========================
# HELPERS
# =========================
def load_model_name(default: str = "sentence-transformers/all-mpnet-base-v2") -> str:
    """
    Try to read paraphraser.model_name from configs/config.yaml.
    Fall back to default if not found.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        paraphraser_cfg = cfg.get("paraphraser", {})
        model_name = paraphraser_cfg.get(
            "model_name") or paraphraser_cfg.get("model")
        if model_name:
            print(f"[INFO] Using paraphraser model from config: {model_name}")
            return model_name

    print(f"[INFO] Using default paraphraser model: {default}")
    return default


def generate_candidates(base_query: str):
    """
    Template-based candidates for different semantic levels.
    Actual binning is decided by cosine similarity.
    """
    q = base_query

    high = [
        f"{q}",
        f"{q} official documentation",
        f"{q} detailed guide",
        f"{q} docs and tutorials",
    ]

    medium = [
        f"step-by-step guide for {q}",
        f"{q} for beginners",
        f"{q} reference and best practices",
        f"how to get started with {q}",
        f"{q} overview and usage examples",
    ]

    low = [
        f"when should developers rely on {q} in real projects",
        f"deep dive explaining how {q} works end-to-end",
        f"use cases, limitations and integration patterns for {q}",
        f"comparison of {q} with similar cloud services",
        f"best resources to master {q} for production workloads",
    ]

    return (
        [(p, "high") for p in high]
        + [(p, "medium") for p in medium]
        + [(p, "low") for p in low]
    )


def bin_similarity(score: float):
    """Map cosine similarity → similarity_bin label."""
    if score >= HIGH_MIN:
        return "high"
    if MED_MIN <= score < MED_MAX:
        return "medium"
    if LOW_MIN <= score < LOW_MAX:
        return "low"
    return None


def pick_base_queries(df: pd.DataFrame) -> pd.Series:
    """
    Pick the correct column from base_queries.csv.
    """
    for col in ["base_query", "seed_query", "query"]:
        if col in df.columns:
            print(f"[INFO] Using '{col}' as base query column")
            return df[col].astype(str)

    first_col = df.columns[0]
    print(
        f"[WARN] 'base_query' not found; using first column '{first_col}' as base query column")
    return df[first_col].astype(str)


# =========================
# MAIN
# =========================
def main():
    model_name = load_model_name()
    model = SentenceTransformer(model_name)

    base_df = pd.read_csv("data/queries/base_queries.csv")
    base_queries = pick_base_queries(base_df)

    rows = []

    for seed in base_queries:
        seed = seed.strip()
        if not seed:
            continue

        seed_emb = model.encode(seed, convert_to_tensor=True)
        seen = set()

        for text, tmpl_level in generate_candidates(seed):
            text = text.strip()
            if not text or text in seen:
                continue
            seen.add(text)

            para_emb = model.encode(text, convert_to_tensor=True)
            sim = float(util.cos_sim(seed_emb, para_emb).item())
            sim_bin = bin_similarity(sim)

            if sim_bin is None:
                continue

            rows.append(
                {
                    "base_query": seed,
                    "paraphrase_query": text,
                    "cosine_similarity": round(sim, 4),
                    "similarity_bin": sim_bin,
                    "template_level": tmpl_level,
                }
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"[OK] Wrote {len(out_df)} paraphrases → {OUT_PATH}")


if __name__ == "__main__":
    main()
