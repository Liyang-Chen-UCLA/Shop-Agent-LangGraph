from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ...domain.criteria import SchemaModel


class EvalGroup(SchemaModel):
    status: Literal["match", "uncertain", "independent"]
    left_ids: list[str]
    right_ids: list[str]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape(self) -> EvalGroup:
        if not self.reason.strip():
            raise ValueError("reason must be nonblank")
        if len(self.left_ids) != len(set(self.left_ids)) or len(self.right_ids) != len(
            set(self.right_ids)
        ):
            raise ValueError("source IDs must be unique within a group")
        if self.status in {"match", "uncertain"}:
            if not self.left_ids or not self.right_ids:
                raise ValueError(f"{self.status} groups require sources from both sides")
        elif bool(self.left_ids) == bool(self.right_ids):
            raise ValueError("independent groups require sources from exactly one side")
        return self


class EvalReport(SchemaModel):
    groups: list[EvalGroup]
