# src/pirpb/metrics.py
from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# ----------------- helpers -----------------


def load_cfg(path: Path) -> dict:
    if not path.exists():
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def norm(s: str) -> str:
    """case/whitespace-insensitive canonical form"""
    return " ".join(str(s).strip().split()).lower() if isinstance(s, str) else ""


def rbo(list1: List[str], list2: List[str], p: float = 0.9) -> float:
    """Extrapolated Rank-Biased Overlap (Webber et al., 2010). Returns [0,1]."""
    if not list1 and not list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    if not (0 < p < 1):
        raise ValueError("p must be in (0,1)")

    # de-dupe preserving order
    A, B = [], []
    seenA, seenB = set(), set()
    for x in list1:
        if x not in seenA:
            seenA.add(x)
            A.append(x)
    for x in list2:
        if x not in seenB:
            seenB.add(x)
            B.append(x)

    k = max(len(A), len(B))
    seenA, seenB = set(), set()
    overlap = 0
    summ = 0.0
    A_d = 0.0

    for d in range(1, k + 1):
        if d <= len(A):
            a = A[d-1]
            seenA.add(a)
            if a in seenB:
                overlap += 1
        if d <= len(B):
            b = B[d-1]
            if b not in seenB:
                seenB.add(b)
                if b in seenA:
                    overlap += 1
        A_d = overlap / d
        summ += A_d * (p ** d)

    rbo_ext = ((1 - p) / p) * summ + A_d * (p ** k)
    return max(0.0, min(1.0, float(rbo_ext)))


def pir_from_rbo(r: float) -> float:
    return 1.0 - float(r) if r is not None and not math.isnan(r) else math.nan

# ----------------- main compute -----------------


def compute_pir():
    cfg = load_cfg(Path("configs/config.yaml"))
    serp_dir = Path(cfg.get("io", {}).get("serp_dir", "serp"))
    results_dir = Path(cfg.get("io", {}).get("results_dir", "results"))
    logs_dir = Path(cfg.get("io", {}).get("logs_dir", "logs"))
    top_k = int(cfg.get("search", {}).get("top_k", 10))
    p = float(cfg.get("metrics", {}).get("rbo_p", 0.9))

    serp_csv = serp_dir / "serp_google.csv"
    paraphrases_csv = Path("data/paraphrases/paraphrases.csv")
    out_csv = results_dir / "pir_google.csv"
    hist_path = results_dir / "pir_hist_google.png"
    log_path = logs_dir / "stage_d_metrics.log"

    # load SERPs and paraphrases
    serp = pd.read_csv(serp_csv)
    para = pd.read_csv(paraphrases_csv)

    serp.columns = [c.lower() for c in serp.columns]
    para.columns = [c.lower() for c in para.columns]

    required_serp = {"query", "rank", "url"}
    if not required_serp.issubset(set(serp.columns)):
        raise ValueError(
            f"serp_google.csv must have columns {required_serp}, found {serp.columns.tolist()}")

    serp["rank"] = pd.to_numeric(serp["rank"], errors="coerce")
    serp = serp.dropna(subset=["rank", "url", "query"]).copy()
    serp["rank"] = serp["rank"].astype(int)

    # canonical query text to improve matching
    serp["query_norm"] = serp["query"].map(norm)
    para["query_norm"] = para["query"].map(norm)
    para["paraphrase_norm"] = para["paraphrase"].map(norm)

    # build fast lookup: {query_norm -> topK url list}
    urls_by_q: Dict[str, List[str]] = {}
    for qn, grp in serp.sort_values("rank").groupby("query_norm"):
        urls_by_q[qn] = grp.head(top_k)["url"].dropna().astype(str).tolist()

    rows = []
    missing_seed, missing_para, short_lists = 0, 0, 0

    for _, r in para.iterrows():
        q_seed_n = r["query_norm"]
        q_para_n = r["paraphrase_norm"]
        urls_seed = urls_by_q.get(q_seed_n, [])
        urls_para = urls_by_q.get(q_para_n, [])
        if not urls_seed:
            missing_seed += 1
            continue
        if not urls_para:
            missing_para += 1
            continue
        # for pilot, allow even small lists (>=1), tighten later if needed
        if len(urls_seed) < 1 or len(urls_para) < 1:
            short_lists += 1
            continue
        score = rbo(urls_seed, urls_para, p)
        rows.append({
            "seed_query": r["query"],
            "paraphrase": r["paraphrase"],
            "k_used": min(top_k, max(len(urls_seed), len(urls_para))),
            "p": p,
            "rbo": score,
            "pir": pir_from_rbo(score),
        })

    # write outputs
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    mean_pir = float(df["pir"].mean()) if not df.empty else float("nan")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=== Stage D (metrics) ===\n")
        f.write(f"pairs_scored={len(df)}\n")
        f.write(f"missing_seed={missing_seed}\n")
        f.write(f"missing_para={missing_para}\n")
        f.write(f"short_lists={short_lists}\n")
        f.write(f"mean_pir={mean_pir}\n\n")

    if not df.empty:
        plt.figure()
        df["pir"].plot(kind="hist", bins=20, edgecolor="black")
        plt.xlabel("PIR (1 - RBO@K)")
        plt.ylabel("Count")
        plt.title("Distribution of PIR (Google CSE)")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=160)
        plt.close()

    print(f"[Stage D] Pairs scored: {len(df)}")
    print(
        f"[Stage D] missing_seed={missing_seed} | missing_para={missing_para} | short_lists={short_lists}")
    print(f"[Stage D] Mean PIR = {mean_pir:.4f}" if not math.isnan(
        mean_pir) else "[Stage D] Mean PIR = NaN")


if __name__ == "__main__":
    compute_pir()
