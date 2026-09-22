from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from ...domain.criteria import (
    Attribute,
    BooleanCriterion,
    CategoricalCriterion,
    Criterion,
    NumericCriterion,
    SchemaModel,
)


CRITERION_TYPES = (NumericCriterion, BooleanCriterion, CategoricalCriterion)


class MatchDecision(SchemaModel):
    left_id: str
    right_id: str
    kind: Literal["criterion", "attribute"]
    item: Criterion | Attribute

    @model_validator(mode="after")
    def validate_item_kind(self) -> MatchDecision:
        item_is_criterion = isinstance(self.item, CRITERION_TYPES)
        if (self.kind == "criterion") != item_is_criterion:
            raise ValueError(f"kind {self.kind!r} does not match the submitted item type")
        return self


class MatchBatch(SchemaModel):
    matches: list[MatchDecision]


class IndependentBatch(SchemaModel):
    item_ids: list[str]


class DiffRequest(SchemaModel):
    left_id: str
    right_id: str
