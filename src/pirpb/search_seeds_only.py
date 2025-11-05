from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

API_URL = "https://www.googleapis.com/customsearch/v1"

def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _flatten_items(items: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    rows = []
    for i, it in enumerate(items or [], start=1):
        rows.append({
            "qid": f"seed::{hash(query) & 0xfffffff}",
            "query": query,
            "rank": i,
            "title": it.get("title", ""),
            "url": it.get("link", ""),
            "snippet": it.get("snippet", "")
        })
    return rows

def _load_serp_dir_from_yaml(cfg_path: Path) -> Path:
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Expect structure: io: { serp_dir: "serp" }
    io = cfg.get("io", {})
    serp_dir = io.get("serp_dir", "serp")
    return Path(serp_dir)

def main() -> int:
    load_dotenv()

    api_key, cx = os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CX")
    if not api_key or not cx:
        print("Missing GOOGLE_API_KEY or GOOGLE_CX in .env", file=sys.stderr)
        return 2

    serp_dir = _load_serp_dir_from_yaml(Path("configs/config.yaml"))
    cache_dir = serp_dir / "cache_google"
    out_csv = serp_dir / "serp_google.csv"
    seeds_csv = Path("data/queries/base_queries.csv")

    _safe_mkdir(serp_dir)
    _safe_mkdir(cache_dir)

    if not seeds_csv.exists():
        print(f"Seeds file not found: {seeds_csv}", file=sys.stderr)
        return 3

    seeds = pd.read_csv(seeds_csv)
    seeds.columns = [c.lower() for c in seeds.columns]
    if "query" not in seeds.columns:
        print("base_queries.csv must have a 'query' column.", file=sys.stderr)
        return 3

    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        existing.columns = [c.lower() for c in existing.columns]
    else:
        existing = pd.DataFrame(columns=["qid","query","rank","title","url","snippet"])

    all_new_rows: List[Dict[str, Any]] = []
    for q in seeds["query"].astype(str).tolist():
        cache_path = cache_dir / f"google_seed_{abs(hash(q)) & 0xfffffff}.json"
        data = None

        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                data = None

        if data is None:
            params = {"key": api_key, "cx": cx, "q": q, "num": 10}
            r = requests.get(API_URL, params=params, timeout=30)
            if r.status_code != 200:
                print(f"[WARN] Google CSE {r.status_code} for '{q}': {r.text[:200]}", file=sys.stderr)
                time.sleep(1.0)
                continue
            data = r.json()
            cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.3)

        rows = _flatten_items(data.get("items", []), q)
        all_new_rows.extend(rows)

    if not all_new_rows:
        print("[INFO] No new seed results fetched.")
        return 0

    new_df = pd.DataFrame(all_new_rows)
    merged = pd.concat([existing, new_df], ignore_index=True)
    merged.drop_duplicates(subset=["query","url","rank"], inplace=True)
    merged.sort_values(by=["query","rank"], inplace=True)
    merged.to_csv(out_csv, index=False)
    print(f"[OK] Updated {out_csv} with seed SERPs.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
