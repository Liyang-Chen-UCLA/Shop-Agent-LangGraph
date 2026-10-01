from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

PACKAGED_PRODUCTS_PATH = Path(__file__).resolve().parents[1] / "data" / "products.parquet"
PRODUCT_COLUMNS = ["item_id", "category", "rank", "title", "context_text"]
OPTIONAL_COLUMNS = ["ocr_pages", "shop", "record_id", "snapshot_id", "context_sha256"]


def products_path() -> Path:
    configured = os.getenv("MARKET_DATASET_PATH")
    return Path(configured) if configured else PACKAGED_PRODUCTS_PATH


@lru_cache(maxsize=1)
def load_products() -> tuple[dict[str, Any], ...]:
    path = products_path()
    available = parquet.read_schema(path).names
    table = parquet.read_table(path, columns=PRODUCT_COLUMNS + [
        name for name in OPTIONAL_COLUMNS if name in available
    ])
    return tuple(table.to_pylist())


def search_product_ids(query: str, limit: int = 10) -> list[str]:
    if not query.strip() or limit < 1:
        raise ValueError("query must be nonempty and limit must be positive")
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
    return get_product(item_id)["context_text"]


def get_product(item_id: str) -> dict[str, Any]:
    for row in load_products():
        if row["item_id"] == str(item_id):
            return row
    raise ValueError(f"unknown market item ID: {item_id}")


def get_product_pages(item_id: str) -> list[dict[str, Any]]:
    """Expose source pages without generating or rewriting OCR evidence."""
    row = get_product(item_id)
    if row.get("ocr_pages"):
        return [{key: page.get(key) for key in (
            "page_id", "page_index", "source_image_paths", "text", "text_sha256"
        )} for page in row["ocr_pages"]]
    # Keep custom/legacy datasets with only context_text usable.
    matches = list(re.finditer(r"^## OCR Page (\d+)\s*$", row["context_text"], re.M))
    if not matches:
        return [dict(page_id="page-0001", page_index=1, source_image_paths=[],
                     text=row["context_text"], text_sha256=None)]
    return [dict(page_id=f"page-{int(match[1]):04d}", page_index=int(match[1]),
                 source_image_paths=[], text=row["context_text"][match.end():
                     matches[i + 1].start() if i + 1 < len(matches) else None].strip(),
                 text_sha256=None) for i, match in enumerate(matches)]


def get_product_summary(item_id: str) -> dict[str, Any]:
    row = get_product(item_id)
    pages = get_product_pages(item_id)
    return {
        **{key: row[key] for key in PRODUCT_COLUMNS if key != "context_text"},
        **{key: row[key] for key in OPTIONAL_COLUMNS if key != "ocr_pages" and key in row},
        "page_count": len(pages),
        "pages": [dict(page_id=page["page_id"], page_index=page["page_index"],
                       character_count=len(page["text"]),
                       preview=page["text"][:160],
                       preview_truncated=len(page["text"]) > 160) for page in pages],
    }


def read_product_pages(item_id: str, page_ids: list[str]) -> dict[str, Any]:
    if not page_ids or len(page_ids) != len(set(page_ids)):
        raise ValueError("provide one or more distinct page IDs")
    pages = {page["page_id"]: page for page in get_product_pages(item_id)}
    missing = set(page_ids) - pages.keys()
    if missing:
        raise ValueError(f"unknown OCR page IDs: {sorted(missing)}")
    return dict(item_id=str(item_id), pages=[pages[page_id] for page_id in page_ids])
