from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LargerBetter(SchemaModel):
    type: Literal["larger_better"]


class SmallerBetter(SchemaModel):
    type: Literal["smaller_better"]


class TargetRange(SchemaModel):
    type: Literal["target_range"]
    unit: str


NumericDirection: TypeAlias = Annotated[
    LargerBetter | SmallerBetter | TargetRange,
    Field(discriminator="type"),
]


class TrueBetter(SchemaModel):
    type: Literal["true_better"]


class FalseBetter(SchemaModel):
    type: Literal["false_better"]


BooleanDirection: TypeAlias = Annotated[
    TrueBetter | FalseBetter,
    Field(discriminator="type"),
]


class TotalOrder(SchemaModel):
    type: Literal["total_order"]
    order: list[str]


class PartialOrder(SchemaModel):
    type: Literal["partial_order"]
    better_than: list[tuple[str, str]]


class PreferredSet(SchemaModel):
    type: Literal["preferred_set"]
    values: list[str]


CategoricalDirection: TypeAlias = Annotated[
    TotalOrder | PartialOrder | PreferredSet,
    Field(discriminator="type"),
]


class CommonItem(SchemaModel):
    id: str
    name: str
    description: str
    aliases: list[str]


class NumericCriterion(CommonItem):
    type: Literal["numeric"]
    units: list[str]
    formula: str | None = None
    direction: NumericDirection


class BooleanCriterion(CommonItem):
    type: Literal["boolean"]
    direction: BooleanDirection


class CategoricalCriterion(CommonItem):
    type: Literal["categorical"]
    values: list[str]
    value_domain: Literal["open", "closed"]
    direction: CategoricalDirection


Criterion: TypeAlias = Annotated[
    NumericCriterion | BooleanCriterion | CategoricalCriterion,
    Field(discriminator="type"),
]


class NumericAttribute(CommonItem):
    type: Literal["numeric"]
    units: list[str]
    formula: str | None = None


class BooleanAttribute(CommonItem):
    type: Literal["boolean"]


class CategoricalAttribute(CommonItem):
    type: Literal["categorical"]
    values: list[str]
    value_domain: Literal["open", "closed"]


Attribute: TypeAlias = Annotated[
    NumericAttribute | BooleanAttribute | CategoricalAttribute,
    Field(discriminator="type"),
]


class CriteriaAttributeSet(SchemaModel):
    source_item_ids: list[str]
    criteria: list[Criterion]
    attributes: list[Attribute]
