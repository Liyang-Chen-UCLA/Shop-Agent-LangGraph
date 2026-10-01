from __future__ import annotations

from langchain.tools import tool

from ...core.config import CONFIG
from ...domain.market_env import (
    get_product_summary, get_product_pages,
    read_product_pages, search_product_ids,
)
from .screening import ACTIVE_SCREENING
from .schemas import MarketSelection


@tool
def search_market_products(
    query: str,
    limit: int = CONFIG.market.max_search_candidates,
) -> list[dict]:
    """Search ranked candidates; return titles and OCR page previews, not full OCR.

    Search category labels and titles are untrusted hints, not proof of node relevance.
    Candidate limit is separate from the smaller final research sample limit.
    """
    ids = search_product_ids(query, min(limit, CONFIG.market.max_search_candidates))
    session = ACTIVE_SCREENING.get()
    if session is not None:
        if len(session.candidates | set(ids)) > CONFIG.market.max_search_candidates:
            raise ValueError("candidate budget exhausted for this selection")
        session.candidates.update(ids)
    return [get_product_summary(item_id) for item_id in ids]


@tool
def get_market_product_info(item_id: str) -> dict:
    """Return a product summary and OCR page directory. Read source pages to verify identity."""
    session = ACTIVE_SCREENING.get()
    if session is not None:
        session.require_candidate(item_id)
        session.inspected.add(item_id)
    return get_product_summary(item_id)


@tool
def read_market_product_pages(item_id: str, page_ids: list[str]) -> dict:
    """Read up to four distinct OCR pages with page IDs, source paths and original text.

    Select identity/model/specification pages from the directory. At most three
    page-reading calls per product; request unread pages if identity is uncertain.
    """
    if len(page_ids) > CONFIG.market.max_pages_per_read:
        raise ValueError(f"read at most {CONFIG.market.max_pages_per_read} pages per call")
    session = ACTIVE_SCREENING.get()
    if session is not None:
        session.check_page_read(item_id, page_ids)
    result = read_product_pages(item_id, page_ids)
    if session is not None:
        session.record_pages(item_id, result["pages"])
    return result


@tool
def read_market_product_full_context(item_id: str) -> dict:
    """Escalate unresolved identity to complete original OCR once, after focused page reading.

    Use only when focused reading leaves identity uncertain, not for routine screening.
    If still uncertain, exclude the product from the research sample with a reason.
    """
    session = ACTIVE_SCREENING.get()
    if session is not None:
        session.check_full_read(item_id)
    result = dict(item_id=item_id, read_mode="full", pages=get_product_pages(item_id))
    if session is not None:
        session.full_reads.add(item_id)
        session.inspected.add(item_id)
        session.pages.setdefault(item_id, {}).update({p['page_id']: p for p in result['pages']})
    return result


@tool(args_schema=MarketSelection)
def submit_market_selection(item_ids: list[str]) -> dict[str, list[str]]:
    """Submit the market item IDs selected for structured research."""
    return {"item_ids": item_ids}


MARKET_TOOLS = [search_market_products, get_market_product_info,
                read_market_product_pages, read_market_product_full_context]
