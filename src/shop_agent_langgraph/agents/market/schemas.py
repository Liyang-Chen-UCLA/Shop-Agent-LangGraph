from __future__ import annotations

from pydantic import Field

from ...core.config import CONFIG
from ...domain.criteria import Attribute, Criterion, SchemaModel


class MarketSelection(SchemaModel):
    item_ids: list[str] = Field(
        min_length=1,
        max_length=CONFIG.market.max_search_products,
    )


class MarketResult(SchemaModel):
    query: str
    item_ids: list[str]
    criteria: list[Criterion]
    attributes: list[Attribute]
