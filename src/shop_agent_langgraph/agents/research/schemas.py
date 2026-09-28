from __future__ import annotations

from ...domain.criteria import SchemaModel


class Evidence(SchemaModel):
    name: str
    value: str | None = None
    unit: str | None = None
    qualifier: str | None = None
    source_text: str


class ResearchResult(SchemaModel):
    item_id: str
    evidence: list[Evidence]
