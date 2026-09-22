from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ...core.config import CONFIG
from ...domain.criteria import (
    Attribute,
    BooleanCriterion,
    CategoricalCriterion,
    CriteriaAttributeSet,
    Criterion,
    NumericCriterion,
    SchemaModel,
)


CRITERION_TYPES = (NumericCriterion, BooleanCriterion, CategoricalCriterion)


class MarketSelection(SchemaModel):
    item_ids: list[str] = Field(
        min_length=1,
        max_length=CONFIG.market.max_search_products,
    )


class CanonicalOutput(SchemaModel):
    kind: Literal["criterion", "attribute"]
    item: Criterion | Attribute

    @model_validator(mode="after")
    def check_kind(self) -> CanonicalOutput:
        item_is_criterion = isinstance(self.item, CRITERION_TYPES)
        if (self.kind == "criterion") != item_is_criterion:
            raise ValueError("kind must agree with the item's criterion/attribute schema")
        return self


class MatchDecision(CanonicalOutput):
    left_id: str
    right_id: str


class MatchBatch(SchemaModel):
    matches: list[MatchDecision] = Field(min_length=1)


class IndependentBatch(SchemaModel):
    item_ids: list[str] = Field(min_length=1)


class ResolvedOutput(CanonicalOutput):
    alignment: Literal["match", "independent"]
    source_refs: list[str] = Field(min_length=1)
    preserved_aspects: list[str] = Field(min_length=1)


class PartialResolution(SchemaModel):
    group_id: str = Field(min_length=1)
    left_ids: list[str] = Field(min_length=1)
    right_ids: list[str] = Field(min_length=1)
    relation: Literal["left_more_specific", "right_more_specific", "overlap"]
    reason: str = Field(min_length=1)
    status: Literal["resolved", "needs_review"]
    outputs: list[ResolvedOutput]
    missing_evidence: list[str]

    @model_validator(mode="after")
    def check_status(self) -> PartialResolution:
        if not self.group_id.strip() or not self.reason.strip():
            raise ValueError("group_id and reason must be nonblank")
        if self.status == "resolved":
            if not self.outputs or self.missing_evidence:
                raise ValueError("resolved groups require outputs and no missing_evidence")
        elif self.outputs or not self.missing_evidence:
            raise ValueError("needs_review groups require missing_evidence and no outputs")
        elif any(not evidence.strip() for evidence in self.missing_evidence):
            raise ValueError("missing_evidence entries must be nonblank")
        return self


class PartialResolutionBatch(SchemaModel):
    resolutions: list[PartialResolution] = Field(min_length=1)


class FinalizeAggregation(SchemaModel):
    """The runtime assembles the collection from accepted aggregation operations."""


class PendingAggregation(SchemaModel):
    group_id: str
    left_source_item_ids: list[str]
    right_source_item_ids: list[str]
    left_ids: list[str]
    right_ids: list[str]
    reason: str
    missing_evidence: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_explanation(self) -> PendingAggregation:
        if not self.group_id.strip() or not self.reason.strip():
            raise ValueError("pending group_id and reason must be nonblank")
        if any(not evidence.strip() for evidence in self.missing_evidence):
            raise ValueError("pending missing_evidence entries must be nonblank")
        return self


class MarketAggregationOutcome(SchemaModel):
    status: Literal["completed", "pending"]
    collection: CriteriaAttributeSet | None = None
    pending_groups: list[PendingAggregation] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_status(self) -> MarketAggregationOutcome:
        if self.status == "completed":
            if self.collection is None or self.pending_groups or self.unresolved:
                raise ValueError("completed aggregation requires only a collection")
        elif self.collection is not None or not self.pending_groups:
            raise ValueError("pending aggregation requires pending_groups and no collection")
        return self


class MarketResult(SchemaModel):
    query: str
    item_ids: list[str]
    status: Literal["completed", "pending"]
    criteria: list[Criterion]
    attributes: list[Attribute]
    pending_groups: list[PendingAggregation] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_status(self) -> MarketResult:
        if self.status == "completed" and self.pending_groups:
            raise ValueError("completed market results cannot contain pending groups")
        if self.status == "pending":
            if not self.pending_groups or self.criteria or self.attributes:
                raise ValueError(
                    "pending market results require pending_groups and no final collection"
                )
        return self
