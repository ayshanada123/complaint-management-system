"""
scripts/prepare_data.py
Turn the raw CFPB Consumer Complaint Database CSV into a clean, balanced
training file with the columns: category, text.

Download the raw data (free, US government, public domain):
    https://www.consumerfinance.gov/data-research/consumer-complaints/
    (also mirrored on Kaggle as "Consumer Complaint Database")

Usage:
    python scripts/prepare_data.py data/raw/complaints.csv --per-class 4000

The raw file is large, so it is read in chunks and only the two needed
columns are kept.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLEAN_DATA_PATH  # noqa: E402
from textutils import clean_text  # noqa: E402


def normalise_column(name: str) -> str:
    """'Consumer complaint narrative' -> 'consumer_complaint_narrative'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def map_product(product: str):
    """Map a CFPB 'Product' value to one of our categories (or None)."""
    p = str(product).lower()
    if "credit reporting" in p or "credit repair" in p:
        return "Credit Reporting"
    if "debt collection" in p:
        return "Debt Collection"
    if "mortgage" in p:
        return "Mortgage"
    if "credit card" in p or "prepaid card" in p:
        return "Credit & Prepaid Cards"
    if "checking or savings" in p or "bank account" in p:
        return "Bank Accounts & Services"
    if any(k in p for k in ("student loan", "vehicle loan", "consumer loan",
                            "payday", "title loan", "personal loan")):
        return "Loans"
    if any(k in p for k in ("money transfer", "virtual currency",
                            "money service")):
        return "Money Transfers"
    return None


def cap_per_class(df, n, seed=42):
    """Keep at most n randomly chosen rows for each category."""
    groups = [g.sample(min(len(g), n), random_state=seed)
              for _, g in df.groupby("category")]
    return pd.concat(groups, ignore_index=True) if groups else df


def prepare(raw_csv, per_class=4000, min_words=8, seed=42, chunksize=200_000):
    wanted = {"product", "consumer_complaint_narrative"}
    parts = []

    reader = pd.read_csv(
        raw_csv,
        usecols=lambda c: normalise_column(c) in wanted,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk.columns = [normalise_column(c) for c in chunk.columns]
        chunk = chunk.dropna(subset=["consumer_complaint_narrative"])
        chunk["category"] = chunk["product"].map(map_product)
        chunk = chunk.dropna(subset=["category"])
        chunk["text"] = chunk["consumer_complaint_narrative"].map(clean_text)
        chunk = chunk[chunk["text"].str.split().str.len() >= min_words]
        if chunk.empty:
            continue
        # Keep memory bounded: at most per_class rows per class per chunk.
        parts.append(cap_per_class(chunk[["category", "text"]], per_class, seed))

    if not parts:
        raise SystemExit("No usable rows found. Check the CSV columns.")

    data = pd.concat(parts, ignore_index=True)
    data = data.drop_duplicates(subset="text")
    data = cap_per_class(data, per_class, seed)
    return data.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_csv", help="Path to the raw CFPB complaints CSV")
    ap.add_argument("--per-class", type=int, default=4000,
                    help="Max complaints kept per category (default 4000)")
    ap.add_argument("--out", default=str(CLEAN_DATA_PATH))
    args = ap.parse_args()

    data = prepare(args.raw_csv, per_class=args.per_class)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)

    print(f"Saved {len(data):,} complaints to {args.out}")
    print(data["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
