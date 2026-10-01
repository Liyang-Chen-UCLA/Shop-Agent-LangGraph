from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MarketConfig:
    max_search_products: int = 4
    max_search_candidates: int = 20
    max_page_reads_per_product: int = 3
    max_pages_per_read: int = 4
    max_aggregation_attempts: int = 8


@dataclass(frozen=True, slots=True)
class AppConfig:
    market: MarketConfig = field(default_factory=MarketConfig)


CONFIG = AppConfig()
