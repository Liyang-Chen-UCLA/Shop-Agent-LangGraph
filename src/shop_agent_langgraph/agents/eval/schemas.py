from __future__ import annotations
from typing import Any, Literal
from pydantic import Field, field_validator
from ...domain.criteria import SchemaModel


class EvalGroup(SchemaModel):
    gold_ids: list[str] = Field(min_length=1)
    actual_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class UnmatchedItem(SchemaModel):
    ref: str
    reason: str = Field(min_length=1)


class MatchingReport(SchemaModel):
    match: list[EvalGroup]
    missing: list[UnmatchedItem]
    extra: list[UnmatchedItem]


class FieldScore(SchemaModel):
    field: Literal["kind", "type", "units", "formula", "direction", "values",
                   "value_domain", "aliases", "granularity"]
    reason: str = Field(min_length=1)
    score: Literal[0, 1]

    @field_validator("score", mode="before")
    @classmethod
    def binary_integer(cls, value):
        if type(value) is not int or value not in (0, 1):
            raise ValueError("score must be the integer 0 or 1")
        return value


class ItemReview(SchemaModel):
    gold_ref: str
    fields: list[FieldScore]


class SchemaReview(SchemaModel):
    items: list[ItemReview]


class Coverage(SchemaModel):
    gold_count: int
    actual_count: int
    matched_gold_count: int
    matched_actual_count: int
    missing_count: int
    extra_count: int


class EvalMetrics(Coverage):
    match_group_count: int
    classification_error_count: int
    granularity_error_count: int
    field_scores: dict[str, float | None]
    by_kind: dict[str, Coverage]


class EvalReport(MatchingReport):
    source_item_ids: list[str]
    items: dict[str, dict[str, Any]]
    reviews: list[ItemReview]
    metrics: EvalMetrics
    gold_metadata: dict[str, Any] = Field(default_factory=dict)
