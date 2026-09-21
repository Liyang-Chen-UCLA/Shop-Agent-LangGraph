from __future__ import annotations

from langchain.tools import tool

from ...core.config import CONFIG
from ...domain.market_env import get_product_raw_text, search_product_ids
from .schemas import MarketSelection


@tool
def search_market_products(
    query: str,
    limit: int = CONFIG.market.max_search_products,
) -> list[str]:
    """Search the local Taobao market dataset and return ranked item IDs."""
    return search_product_ids(query, min(limit, CONFIG.market.max_search_products))


@tool
def get_market_product_info(item_id: str) -> str:
    """Return the unmodified raw OCR text for one market item ID."""
    return get_product_raw_text(item_id)


@tool(args_schema=MarketSelection)
def submit_market_selection(item_ids: list[str]) -> dict[str, list[str]]:
    """Submit the market item IDs selected for structured research."""
    return {"item_ids": item_ids}


MARKET_TOOLS = [search_market_products, get_market_product_info]
