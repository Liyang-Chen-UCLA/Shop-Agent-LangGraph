from __future__ import annotations

from collections import Counter
from typing import Any

from langchain.tools import tool

from ...domain.criteria import Attribute, CriteriaAttributeSet, Criterion
from .schemas import DiffRequest, IndependentBatch, MatchBatch, MatchDecision


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _ensure_references_available(
    references: list[str],
    unresolved: dict[str, Criterion | Attribute],
) -> None:
    duplicate_references = _duplicates(references)
    if duplicate_references:
        raise ValueError(f"references must be unique within a batch: {duplicate_references}")

    missing_references = sorted(set(references) - unresolved.keys())
    if missing_references:
        raise ValueError(f"unresolved references do not exist: {missing_references}")


def _ensure_output_ids_available(
    new_items: list[Criterion | Attribute],
    criteria: list[Criterion],
    attributes: list[Attribute],
) -> None:
    output_ids = [item.id for item in [*criteria, *attributes, *new_items]]
    duplicate_ids = _duplicates(output_ids)
    if duplicate_ids:
        raise ValueError(f"canonical output IDs would conflict: {duplicate_ids}")


def build_eval_tools(
    left: CriteriaAttributeSet,
    right: CriteriaAttributeSet,
) -> tuple[
    list[Any],
    dict[str, Criterion | Attribute],
    list[Criterion],
    list[Attribute],
]:
    unresolved: dict[str, Criterion | Attribute] = {}
    kinds: dict[str, str] = {}
    sides: dict[str, str] = {}
    for side, value in (("left", left), ("right", right)):
        source_items = [*value.criteria, *value.attributes]
        duplicate_source_ids = _duplicates([item.id for item in source_items])
        if duplicate_source_ids:
            raise ValueError(
                f"duplicate item IDs in {side} input: {duplicate_source_ids}"
            )

        source_groups = (("criterion", value.criteria), ("attribute", value.attributes))
        for kind, items in source_groups:
            for item in items:
                reference = f"{side}:{kind}:{item.id}"
                unresolved[reference] = item
                kinds[reference] = kind
                sides[reference] = side

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
        decisions = [MatchDecision.model_validate(decision) for decision in matches]
        references = [
            reference
            for decision in decisions
            for reference in (decision.left_id, decision.right_id)
        ]

        # Validate the entire batch before mutating any captured state.
        _ensure_references_available(references, unresolved)
        invalid_sides = [
            f"{decision.left_id} -> {decision.right_id}"
            for decision in decisions
            if sides[decision.left_id] != "left" or sides[decision.right_id] != "right"
        ]
        if invalid_sides:
            raise ValueError(
                "match requires a left reference followed by a right reference: "
                f"{invalid_sides}"
            )
        _ensure_output_ids_available(
            [decision.item for decision in decisions], criteria, attributes
        )

        for reference in references:
            del unresolved[reference]
        criteria.extend(
            decision.item for decision in decisions if decision.kind == "criterion"
        )
        attributes.extend(
            decision.item for decision in decisions if decision.kind == "attribute"
        )
        return {"accepted": len(decisions), "unresolved": list(unresolved)}

    @tool("independent", args_schema=IndependentBatch)
    def independent(item_ids: list[str]) -> dict[str, Any]:
        """Batch-accept unresolved items that represent independent concepts."""
        # Resolve and validate the entire batch before mutating any captured state.
        _ensure_references_available(item_ids, unresolved)
        items = [unresolved[item_id] for item_id in item_ids]
        _ensure_output_ids_available(items, criteria, attributes)

        for item_id in item_ids:
            item = unresolved[item_id]
            if kinds[item_id] == "criterion":
                criteria.append(item)  # type: ignore[arg-type]
            else:
                attributes.append(item)  # type: ignore[arg-type]
        for item_id in item_ids:
            del unresolved[item_id]
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
