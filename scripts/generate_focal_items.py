import csv
from pathlib import Path

BASE_QUERIES_PATH = Path("data/queries/base_queries.csv")
OUTPUT_PATH = Path("data/queries/focal_items.csv")

# Simple default mapping by category
FOCAL_DOMAIN_BY_CATEGORY = {
    "google": "cloud.google.com",
    "microsoft": "learn.microsoft.com",
    "ibm": "cloud.ibm.com",
    "neutral": "",
}


def main():
    if not BASE_QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Base queries CSV not found at {BASE_QUERIES_PATH}")

    rows = []
    with BASE_QUERIES_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["query"]
            category = row["category"]
            focal_domain = FOCAL_DOMAIN_BY_CATEGORY.get(category, "")
            rows.append({
                # IMPORTANT: item_pir.py expects this column name:
                "query": query,
                "category": category,
                "focal_type": "domain" if focal_domain else "",
                "focal_value": focal_domain,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "category", "focal_type", "focal_value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
