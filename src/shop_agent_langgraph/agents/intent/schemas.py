from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints


NormalizedText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class IntentResult(BaseModel):
    """Structured interpretation of one user shopping request."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["create", "update", "remove", "confirm", "switch", "query"]
    category: NormalizedText | None
    criteria_preferences: list[NormalizedText]
    attribute_preferences: list[NormalizedText]
