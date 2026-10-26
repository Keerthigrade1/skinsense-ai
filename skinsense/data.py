"""Loading the sample catalog and reviews.

All products, brands, prices and reviews in data/ are fictional demo data.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_PDF = ROOT / "sample_docs" / "hydra_drench_serum_info.pdf"

SKIN_TYPES = ["dry", "oily", "combination", "normal", "sensitive"]
CONCERNS = ["dryness", "oiliness", "breakouts", "dullness", "uneven tone", "texture", "sensitivity"]
LIST_COLUMNS = ["skin_types", "concerns", "key_ingredients"]


@lru_cache(maxsize=1)
def load_products() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "products.csv")
    for col in LIST_COLUMNS:
        df[col] = df[col].fillna("").map(lambda s: [x.strip() for x in s.split("|") if x.strip()])
    for col in ["fragrance_free", "vegan"]:
        df[col] = df[col].astype(str).str.lower().eq("true")
    return df


@lru_cache(maxsize=1)
def load_reviews() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "reviews.csv")


def categories() -> list[str]:
    return sorted(load_products()["category"].unique())


def product_fact_sheet(row: pd.Series) -> str:
    """Plain-text description of a catalog product built only from catalog fields."""
    return (
        f"{row['name']} by {row['brand']} is a {row['category'].lower()} ({row['texture']}, {row['size_ml']} ml). "
        f"{row['description']} "
        f"Key ingredients: {', '.join(row['key_ingredients'])}. "
        f"Fragrance-free: {'yes' if row['fragrance_free'] else 'no, it contains added fragrance'}. "
        f"Vegan: {'yes' if row['vegan'] else 'no'}. "
        f"Listed skin types: {', '.join(row['skin_types'])}. "
        f"Sample price: ${row['price_usd']:.2f}."
    )
