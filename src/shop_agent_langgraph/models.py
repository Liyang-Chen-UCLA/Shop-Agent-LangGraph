from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaxonomyNode(BaseModel):
    """The public, immutable projection of a taxonomy node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    node_name: str
    node_path: str


class RouteResult(BaseModel):
    """The canonical taxonomy decision for one normalized product name."""

    model_config = ConfigDict(extra="forbid")

    product: str = Field(min_length=1)
    status: Literal["resolved", "ambiguous"]
    resolved_nodes: list[TaxonomyNode]
    candidates: list[TaxonomyNode] = Field(max_length=3)
    children: list[TaxonomyNode]

    @model_validator(mode="after")
    def validate_status_payload(self) -> RouteResult:
        if self.status == "resolved":
            if len(self.resolved_nodes) != 1:
                raise ValueError("resolved results must contain exactly one resolved node")
        elif self.resolved_nodes or self.children:
            raise ValueError("ambiguous results cannot contain resolved nodes or children")
        return self
