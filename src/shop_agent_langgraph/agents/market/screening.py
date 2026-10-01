"""Per-invocation reading limits shared by selection tools, never global history."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from ...core.config import CONFIG


@dataclass
class ScreeningSession:
    candidates: set[str] = field(default_factory=set)
    inspected: set[str] = field(default_factory=set)
    page_reads: dict[str, int] = field(default_factory=dict)
    full_reads: set[str] = field(default_factory=set)
    pages: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def require_candidate(self, item_id: str) -> None:
        if item_id not in self.candidates:
            raise ValueError("product must come from this invocation's search results")

    def check_page_read(self, item_id: str, page_ids: list[str]) -> None:
        self.require_candidate(item_id)
        if item_id in self.full_reads:
            raise ValueError("full OCR already read; use the existing evidence")
        if self.page_reads.get(item_id, 0) >= CONFIG.market.max_page_reads_per_product:
            raise ValueError("page-read budget exhausted; use the one full-context fallback")
        if len(page_ids) > CONFIG.market.max_pages_per_read:
            raise ValueError(f"read at most {CONFIG.market.max_pages_per_read} pages per call")
        if set(page_ids) & self.pages.get(item_id, {}).keys():
            raise ValueError("page already read; request unread pages")

    def record_pages(self, item_id: str, pages: list[dict[str, Any]]) -> None:
        self.inspected.add(item_id)
        self.page_reads[item_id] = self.page_reads.get(item_id, 0) + 1
        self.pages.setdefault(item_id, {}).update({page['page_id']: page for page in pages})

    def check_full_read(self, item_id: str) -> None:
        self.require_candidate(item_id)
        if not self.page_reads.get(item_id):
            raise ValueError("read focused OCR pages before requesting full context")
        if item_id in self.full_reads:
            raise ValueError("full-context fallback can only be used once per product")


ACTIVE_SCREENING: ContextVar[ScreeningSession | None] = ContextVar("market_screening", default=None)


@contextmanager
def screening_session():
    session = ScreeningSession()
    token = ACTIVE_SCREENING.set(session)
    try:
        yield session
    finally:
        ACTIVE_SCREENING.reset(token)
