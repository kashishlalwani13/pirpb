# ==========================================================
# search_duckduckgo.py | Stage C – DuckDuckGo Retrieval
# Reads data/paraphrases/paraphrases.csv, queries DuckDuckGo
# (HTML endpoint), caches raw HTML in serp/cache_duckduckgo/,
# writes normalized SERP to serp/serp_duckduckgo.csv.
#
# This mirrors search_google_cse.py so downstream item_pir.py
# can operate with a different input SERP file.
# ==========================================================
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import requests
import pandas as pd
from bs4 import BeautifulSoup

from .config import load_settings


def ddg_search_html(query: str) -> str:
    """
    Query DuckDuckGo HTML endpoint and return the raw HTML.
    We use the /html endpoint which is lighter and easier to parse.
    """
    url = "https://duckduckgo.com/html/"
    params = {"q": query}
    headers = {
        # Pretend to be a normal browser to avoid weird responses
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.0 Safari/605.1.15"
        )
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def parse_ddg_results(
    html: str, qid: int, query: str, max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Parse DuckDuckGo HTML SERP into a normalized list of results:
    [
        {
            "qid": qid,
            "query": query,
            "rank": j,
            "title": "...",
            "url": "...",
            "snippet": "..."
        },
        ...
    ]
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, Any]] = []

    # DuckDuckGo typical result structure:
    # <div class="result"> ... <a class="result__a" href="...">Title</a> ...
    for j, a in enumerate(soup.select("a.result__a"), start=1):
        if j > max_results:
            break

        title = a.get_text(" ", strip=True)
        url = a.get("href", "")

        # Try to find a nearby snippet
        snippet = ""
        result_div = a.find_parent("div", class_="result")
        if result_div is not None:
            snippet_tag = result_div.select_one(".result__snippet")
            if snippet_tag is not None:
                snippet = snippet_tag.get_text(" ", strip=True)

        rows.append(
            {
                "qid": qid,
                "query": query,
                "rank": j,
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )

    # If nothing parsed, return an "ERROR" row for debugging
    if not rows:
        rows.append(
            {
                "qid": qid,
                "query": query,
                "rank": None,
                "title": None,
                "url": None,
                "snippet": "ERROR: No results parsed from DuckDuckGo HTML.",
            }
        )

    return rows


def main() -> None:
    settings = load_settings()
    in_csv = Path(settings.data_dir) / "paraphrases" / "paraphrases.csv"
    assert in_csv.exists(), f"Missing {in_csv}. Run Stage B paraphraser first."

    serp_dir = Path(settings.serp_dir)
    serp_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = serp_dir / "cache_duckduckgo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    print(f"🦆 [DuckDuckGo] Loaded {len(df)} paraphrased queries")

    all_rows: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        q = str(row["paraphrase"]).strip()
        cache_path = cache_dir / f"query_{i}.html"

        # 1) Load from cache if available
        if cache_path.exists():
            html = cache_path.read_text(encoding="utf-8")
        else:
            try:
                html = ddg_search_html(q)
            except requests.RequestException as e:
                # Catch *any* request-related error (HTTP, connection, timeout, etc.)
                error_info = {
                    "error": str(e),
                    "type": type(e).__name__,
                }
                # Cache error info for debugging
                (cache_dir / f"query_{i}.error.json").write_text(
                    json.dumps(error_info, indent=2),
                    encoding="utf-8",
                )
                all_rows.append(
                    {
                        "qid": i,
                        "query": q,
                        "rank": None,
                        "title": None,
                        "url": None,
                        "snippet": f"ERROR: {error_info}",
                    }
                )
                # Skip to next query instead of crashing whole run
                continue

            cache_path.write_text(html, encoding="utf-8")
            time.sleep(1)  # polite throttle vs DuckDuckGo

        # 2) Parse HTML into structured SERP rows
        rows = parse_ddg_results(html, qid=i, query=q, max_results=10)
        all_rows.extend(rows)

    out_csv = serp_dir / "serp_duckduckgo.csv"
    pd.DataFrame(
        all_rows, columns=["qid", "query", "rank", "title", "url", "snippet"]
    ).to_csv(out_csv, index=False)
    print(
        f"✅ [DuckDuckGo] Saved results to {out_csv} | {len(all_rows)} rows total")


if __name__ == "__main__":
    main()
