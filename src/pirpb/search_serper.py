
# ==========================================================
# search_serper.py | Stage C – Search Retrieval (Serper.dev)
# ----------------------------------------------------------
# Reads paraphrases.csv, calls Serper (Google wrapper),
# caches raw JSON, and writes a flat serp_serper.csv.
# For reproducibility we cache every response to serp/cache/.
# ==========================================================

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any


import requests
import pandas as pd
from dotenv import load_dotenv

# Make sure we can import our config regardless of CWD
from pathlib import Path as _P
import sys as _sys
_SRC = _P(__file__).resolve().parents[1]  # .../src
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from pirpb.config import load_settings  # noqa: E402


def serper_search(query: str, api_key: str) -> Dict[str, Any]:
    """Call Serper.dev Google Search wrapper API and return JSON."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    resp = requests.post(
        "https://google.serper.dev/search",
        headers=headers,
        json={"q": query},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    load_dotenv()  # reads .env in project root
    api_key = os.getenv("SERPER_API_KEY")
    assert api_key, "❌ Missing SERPER_API_KEY in .env"

    settings = load_settings()
    in_csv = Path(settings.data_dir) / "paraphrases" / "paraphrases.csv"
    assert in_csv.exists(), f"Missing {in_csv}. Run Stage B paraphraser first."

    serp_dir = Path(settings.serp_dir)
    cache_dir = serp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    print(f"🔍 Loaded {len(df)} paraphrased queries")

    rows: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        q = str(row["paraphrase"]).strip()
        cache_path = cache_dir / f"query_{i}.json"

        # Use cache if available; otherwise call API and cache
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
        else:
            try:
                data = serper_search(q, api_key)
            except requests.HTTPError as e:
                data = {"error": str(e), "status_code": getattr(
                    e.response, "status_code", None)}
            cache_path.write_text(json.dumps(data, indent=2))
            time.sleep(1)  # polite throttle for free tier

        # Flatten results
        if "error" in data:
            rows.append({
                "qid": i, "query": q, "rank": None,
                "title": None, "url": None,
                "snippet": f"ERROR: {data.get('error')}"
            })
        else:
            for j, item in enumerate(data.get("organic", []), start=1):
                rows.append({
                    "qid": i,
                    "query": q,
                    "rank": j,
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })

    out_csv = serp_dir / "serp_serper.csv"
    pd.DataFrame(rows, columns=[
                 "qid", "query", "rank", "title", "url", "snippet"]).to_csv(out_csv, index=False)
    print(f"Saved SERPER results to {out_csv} | {len(rows)} rows total")


if __name__ == "__main__":
    main()
