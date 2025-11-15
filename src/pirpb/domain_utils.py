"""
domain_utils.py
----------------
Utilities for mapping SERP result URLs to high-level vendor buckets
using configs/domain_map.yaml.

Used by:
- Perturbation Bias (PB) computations
- Any analysis that needs to group results by vendor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import yaml


DEFAULT_BUCKET = "other"


@dataclass
class DomainMap:
    """
    Holds normalized mapping:
      domain -> vendor_bucket
    Example:
      'cloud.google.com' -> 'google'
      'github.com'       -> 'neutral'
    """
    domain_to_vendor: Dict[str, str]

    def get_bucket_for_url(self, url: str) -> str:
        """
        Given a full URL, return its vendor bucket:
          'google', 'microsoft', 'ibm', 'neutral', or 'other'.

        Logic:
        - Extract hostname (no scheme, no path).
        - Normalize to lowercase, strip 'www.'.
        - Try exact match.
        - Try parent domain match (e.g., foo.bar.google.com -> google.com).
        - Fallback to DEFAULT_BUCKET.
        """
        if not url:
            return DEFAULT_BUCKET

        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower().strip()
        except Exception:
            return DEFAULT_BUCKET

        if not host:
            return DEFAULT_BUCKET

        # strip leading 'www.'
        if host.startswith("www."):
            host = host[4:]

        # 1) exact match
        if host in self.domain_to_vendor:
            return self.domain_to_vendor[host]

        # 2) walk up subdomains: a.b.c.com -> b.c.com -> c.com
        parts = host.split(".")
        for i in range(1, len(parts) - 1):
            cand = ".".join(parts[i:])
            if cand in self.domain_to_vendor:
                return self.domain_to_vendor[cand]

        return DEFAULT_BUCKET


def load_domain_map(project_root: Optional[Path] = None) -> DomainMap:
    """
    Load configs/domain_map.yaml and build a DomainMap.

    Expected YAML structure:

        vendors:
          google:
            - cloud.google.com
            - developers.google.com
          microsoft:
            - microsoft.com
          ibm:
            - ibm.com
          neutral:
            - github.com

    Returns:
        DomainMap with domain_to_vendor filled.
    """
    if project_root is None:
        # src/pirpb/domain_utils.py -> parents[2] = repo root
        project_root = Path(__file__).resolve().parents[2]

    cfg_path = project_root / "configs" / "domain_map.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"domain_map.yaml not found at {cfg_path}. "
            "Create it under configs/ before running PB metrics."
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    vendors = raw.get("vendors", {})
    if not isinstance(vendors, dict) or not vendors:
        raise ValueError(
            "domain_map.yaml must define a top-level 'vendors' mapping."
        )

    mapping: Dict[str, str] = {}

    for vendor, domains in vendors.items():
        if not domains:
            continue
        vendor_bucket = str(vendor).strip().lower()
        for d in domains:
            d_str = str(d).strip().lower()
            if not d_str:
                continue
            # strip 'www.' if present
            if d_str.startswith("www."):
                d_str = d_str[4:]
            mapping[d_str] = vendor_bucket

    if not mapping:
        raise ValueError(
            "No valid domain entries found in domain_map.yaml. "
            "Please add domains under each vendor."
        )

    return DomainMap(domain_to_vendor=mapping)


# Small self-test helper (optional)
def _demo():
    dm = load_domain_map()
    samples = [
        "https://cloud.google.com/bigquery/docs",
        "https://www.github.com/tensorflow/tensorflow",
        "https://learn.microsoft.com/en-us/azure/",
        "https://cloud.ibm.com/docs",
        "https://unknownvendor.example.com",
    ]
    for u in samples:
        print(u, "->", dm.get_bucket_for_url(u))


if __name__ == "__main__":
    _demo()
