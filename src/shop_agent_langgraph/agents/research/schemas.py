from __future__ import annotations
from typing import Literal

from ...domain.criteria import SchemaModel


class Evidence(SchemaModel):
    subject: Literal["product", "variant", "accessory", "comparison", "unknown"] = "product"
    claim_kind: Literal["specification", "merchant_claim", "verified"] = "merchant_claim"
    name: str
    value: str | None = None
    unit: str | None = None
    qualifier: str | None = None
    source_text: str


class ResearchResult(SchemaModel):
    relevance: Literal["relevant", "irrelevant", "uncertain"] = "uncertain"
    relevance_reason: str = "Not assessed"
    item_id: str
    evidence: list[Evidence]
