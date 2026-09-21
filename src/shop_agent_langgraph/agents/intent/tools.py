from __future__ import annotations

from typing import Any, Literal

from langchain.tools import tool

from .schemas import IntentResult


@tool(args_schema=IntentResult)
def submit_intent_result(
    action: Literal["create", "update", "remove", "confirm", "switch", "query"],
    category: str | None,
    criteria_preferences: list[str],
    attribute_preferences: list[str],
) -> dict[str, Any]:
    """Submit the parsed user request for runtime validation."""
    return IntentResult(
        action=action,
        category=category,
        criteria_preferences=criteria_preferences,
        attribute_preferences=attribute_preferences,
    ).model_dump()


INTENT_TOOLS = [submit_intent_result]
