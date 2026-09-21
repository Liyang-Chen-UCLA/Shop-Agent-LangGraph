from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

PACKAGED_PRODUCTS_PATH = Path(__file__).resolve().parents[1] / "data" / "products.parquet"
PRODUCT_COLUMNS = ["item_id", "category", "rank", "title", "context_text"]


def products_path() -> Path:
    configured = os.getenv("MARKET_DATASET_PATH")
    return Path(configured) if configured else PACKAGED_PRODUCTS_PATH


@lru_cache(maxsize=1)
def load_products() -> tuple[dict[str, Any], ...]:
    table = parquet.read_table(products_path(), columns=PRODUCT_COLUMNS)
    return tuple(table.to_pylist())


def search_product_ids(query: str, limit: int = 10) -> list[str]:
    normalized_query = query.strip().casefold()
    rows = [
        row
        for row in load_products()
        if normalized_query in row["category"].casefold()
        or normalized_query in row["title"].casefold()
    ]
    rows.sort(key=lambda row: (row["rank"], row["item_id"]))
    return [row["item_id"] for row in rows[:limit]]


def get_product_raw_text(item_id: str) -> str:
    row = next(row for row in load_products() if row["item_id"] == str(item_id))
    return row["context_text"]
