# ==========================================================
# search_google_cse.py | Stage C – Google CSE Retrieval
# Reads data/paraphrases/paraphrases.csv, calls Google Custom Search,
# caches raw JSON in serp/cache_google/, writes serp/serp_google.csv.
# ==========================================================
import os, time, json
from pathlib import Path
from typing import List, Dict, Any
import requests
import pandas as pd
from dotenv import load_dotenv
from pirpb.config import load_settings

def cse_search(query: str, api_key: str, cx: str, num: int = 10) -> Dict[str, Any]:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": query, "num": num}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    # Load keys from project root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CX")
    assert api_key and cx, "❌ Missing GOOGLE_API_KEY or GOOGLE_CX in .env"

    settings = load_settings()
    in_csv = Path(settings.data_dir) / "paraphrases" / "paraphrases.csv"
    assert in_csv.exists(), f"Missing {in_csv}. Run Stage B paraphraser first."

    serp_dir = Path(settings.serp_dir); serp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = serp_dir / "cache_google"; cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    print(f"🔍 Loaded {len(df)} paraphrased queries")

    rows: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        q = str(row["paraphrase"]).strip()
        cache_path = cache_dir / f"query_{i}.json"

        if cache_path.exists():
            data = json.loads(cache_path.read_text())
        else:
            try:
                data = cse_search(q, api_key, cx, num=10)
            except requests.HTTPError as e:
                data = {"error": str(e), "status_code": getattr(e.response, "status_code", None)}
            cache_path.write_text(json.dumps(data, indent=2))
            time.sleep(1)  # polite throttle

        if "error" in data:
            rows.append({"qid": i, "query": q, "rank": None, "title": None, "url": None,
                         "snippet": f"ERROR: {data.get('error')}"})
        else:
            for j, item in enumerate(data.get("items", []), start=1):
                rows.append({
                    "qid": i,
                    "query": q,
                    "rank": j,
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })

    out_csv = serp_dir / "serp_google.csv"
    pd.DataFrame(rows, columns=["qid","query","rank","title","url","snippet"]).to_csv(out_csv, index=False)
    print(f"✅ Saved Google CSE results to {out_csv} | {len(rows)} rows total")

if __name__ == "__main__":
    main()
