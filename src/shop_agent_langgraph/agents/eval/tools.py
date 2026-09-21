from __future__ import annotations

from typing import Any

from langchain.tools import tool

from ...domain.criteria import Attribute, CriteriaAttributeSet, Criterion
from .schemas import DiffRequest, IndependentBatch, MatchBatch, MatchDecision


def build_eval_tools(
    left: CriteriaAttributeSet,
    right: CriteriaAttributeSet,
) -> tuple[list[Any], dict[str, Criterion | Attribute], list[Criterion], list[Attribute]]:
    unresolved: dict[str, Criterion | Attribute] = {}
    kinds: dict[str, str] = {}
    for side, value in (("left", left), ("right", right)):
        for item in value.criteria:
            reference = f"{side}:criterion:{item.id}"
            unresolved[reference] = item
            kinds[reference] = "criterion"
        for item in value.attributes:
            reference = f"{side}:attribute:{item.id}"
            unresolved[reference] = item
            kinds[reference] = "attribute"

    criteria: list[Criterion] = []
    attributes: list[Attribute] = []

    @tool("get_diff", args_schema=DiffRequest)
    def get_diff(left_id: str, right_id: str) -> dict[str, Any]:
        """Return differing fields for two unresolved source-qualified item IDs."""
        left_item = unresolved[left_id].model_dump(mode="json")
        right_item = unresolved[right_id].model_dump(mode="json")
        keys = left_item.keys() | right_item.keys()
        return {
            key: {"left": left_item.get(key), "right": right_item.get(key)}
            for key in keys
            if left_item.get(key) != right_item.get(key)
        }

    @tool("match", args_schema=MatchBatch)
    def match(matches: list[MatchDecision]) -> dict[str, Any]:
        """Batch-merge semantically matching unresolved items."""
        for raw_decision in matches:
            decision = MatchDecision.model_validate(raw_decision)
            unresolved.pop(decision.left_id)
            unresolved.pop(decision.right_id)
            if decision.kind == "criterion":
                criteria.append(decision.item)  # type: ignore[arg-type]
            else:
                attributes.append(decision.item)  # type: ignore[arg-type]
        return {"accepted": len(matches), "unresolved": list(unresolved)}

    @tool("independent", args_schema=IndependentBatch)
    def independent(item_ids: list[str]) -> dict[str, Any]:
        """Batch-accept unresolved items that represent independent concepts."""
        for item_id in item_ids:
            item = unresolved.pop(item_id)
            if kinds[item_id] == "criterion":
                criteria.append(item)  # type: ignore[arg-type]
            else:
                attributes.append(item)  # type: ignore[arg-type]
        return {"accepted": len(item_ids), "unresolved": list(unresolved)}

    @tool("submit", args_schema=CriteriaAttributeSet)
    def submit(
        source_item_ids: list[str],
        criteria: list[Criterion],
        attributes: list[Attribute],
    ) -> dict[str, Any]:
        """Submit the fully merged criteria and attributes."""
        return CriteriaAttributeSet(
            source_item_ids=source_item_ids,
            criteria=criteria,
            attributes=attributes,
        ).model_dump()

    return [get_diff, match, independent, submit], unresolved, criteria, attributes
