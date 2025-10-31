# ==========================================================
# paraphraser.py  |  Stage B - Entity-safe paraphrasing
# ==========================================================
# Generates 1–2 paraphrases per base query without training.
# Preserves vendor/product entities using a simple lock mechanism.
# Filters outputs by cosine similarity (>= threshold from config).
# Output CSV: data/paraphrases/paraphrases.csv
# ----------------------------------------------------------
# AI-ASSIST: Template strategy inspired by standard paraphrase patterns.

from pathlib import Path
import re
import pandas as pd
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from .config import load_settings

# ---------- Helpers ----------


def _protect_entities(text: str, entities: List[str]) -> tuple[str, list[tuple[str, str]]]:
    """
    Replace each entity in `entities` with a placeholder ENT_i to prevent
    templates from altering them. Returns (protected_text, mapping).
    """
    protected_map: list[tuple[str, str]] = []
    tmp = text
    for i, ent in enumerate(entities):
        ent = ent.strip()
        if not ent:
            continue
        placeholder = f"ENT_{i}"
        # case-insensitive exact replacement
        tmp = re.sub(re.escape(ent), placeholder, tmp, flags=re.IGNORECASE)
        protected_map.append((placeholder, ent))
    return tmp, protected_map


def _restore_entities(text: str, protected: list[tuple[str, str]]) -> str:
    out = text
    for placeholder, ent in protected:
        out = out.replace(placeholder, ent)
    return out


def _templates(q: str) -> list[str]:
    """Lightweight templates that keep semantics close."""
    return [
        f"{q} — quick guide",
        f"{q} (overview and examples)",
        f"Where to find {q}",
        f"{q}: beginner-friendly resources",
        f"Best reference for {q}",
    ]


def generate_paraphrases(query: str, lock_entities: list[str], per_query: int = 2) -> list[str]:
    base, protected = _protect_entities(query, lock_entities)
    candidates = _templates(base)
    # Keep order & deduplicate
    seen, unique = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    unique = unique[:max(1, per_query)]
    return [_restore_entities(c, protected) for c in unique]

# ---------- Main ----------


def main():
    settings = load_settings()
    in_csv = Path(settings.data_dir) / "queries" / "base_queries.csv"
    out_csv = Path(settings.data_dir) / "paraphrases" / "paraphrases.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    assert {"category", "query", "lock_entities"}.issubset(df.columns), \
        "CSV must contain category, query, lock_entities"

    model = SentenceTransformer(settings.embed_model)

    rows: list[dict] = []
    for _, row in df.iterrows():
        category = str(row["category"]).strip()
        query = str(row["query"]).strip()
        locks = [x.strip()
                 for x in str(row["lock_entities"]).split("|") if x.strip()]

        # Generate 1–2 paraphrases
        paras = generate_paraphrases(query, locks, per_query=2)

        # Compute cosine(query, paraphrase) and filter
        base_emb = model.encode([query], normalize_embeddings=True)[0]
        for p in paras:
            par_emb = model.encode([p], normalize_embeddings=True)[0]
            cos = float(util.cos_sim(base_emb, par_emb))
            if cos >= settings.cosine_min:  # guardrail from spec
                rows.append({
                    "category": category,
                    "query": query,
                    "paraphrase": p,
                    "cosine": round(cos, 4)
                })

    out_df = pd.DataFrame(
        rows, columns=["category", "query", "paraphrase", "cosine"])
    out_df.to_csv(out_csv, index=False)
    print(
        f"Saved paraphrases to {out_csv} | kept {len(out_df)} rows (cos >= {settings.cosine_min})")


if __name__ == "__main__":
    main()
