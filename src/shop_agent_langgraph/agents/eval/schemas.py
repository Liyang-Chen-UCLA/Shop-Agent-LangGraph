from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ...domain.criteria import (
    Attribute,
    BooleanCriterion,
    CategoricalCriterion,
    Criterion,
    NumericCriterion,
    SchemaModel,
)


CRITERION_TYPES = (NumericCriterion, BooleanCriterion, CategoricalCriterion)


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
        if self.status == "resolved":
            if not self.outputs or self.missing_evidence:
                raise ValueError("resolved groups require outputs and no missing_evidence")
        elif self.outputs or not self.missing_evidence:
            raise ValueError("needs_review groups require missing_evidence and no outputs")
        return self


class PartialResolutionBatch(SchemaModel):
    resolutions: list[PartialResolution] = Field(min_length=1)


class FinalizeRequest(SchemaModel):
    """No model-authored collection: the runtime assembles the result."""
