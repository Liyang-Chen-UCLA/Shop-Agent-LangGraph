from __future__ import annotations

from ...domain.criteria import Attribute, Criterion, SchemaModel


class ResearchResult(SchemaModel):
    item_id: str
    criteria: list[Criterion]
    attributes: list[Attribute]
