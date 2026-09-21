from __future__ import annotations

from typing import Literal

from ...domain.criteria import Attribute, Criterion, SchemaModel


class MatchDecision(SchemaModel):
    left_id: str
    right_id: str
    kind: Literal["criterion", "attribute"]
    item: Criterion | Attribute


class MatchBatch(SchemaModel):
    matches: list[MatchDecision]


class IndependentBatch(SchemaModel):
    item_ids: list[str]


class DiffRequest(SchemaModel):
    left_id: str
    right_id: str
