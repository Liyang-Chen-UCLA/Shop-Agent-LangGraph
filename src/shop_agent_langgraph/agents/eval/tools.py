from __future__ import annotations

from collections import Counter

from langchain.tools import tool

from ...domain.criteria import Attribute, CriteriaAttributeSet, Criterion
from .schemas import EvalGroup, EvalReport


def source_references(
    left: CriteriaAttributeSet,
    right: CriteriaAttributeSet,
) -> tuple[dict[str, Criterion | Attribute], set[str], set[str]]:
    items: dict[str, Criterion | Attribute] = {}
    sides: dict[str, set[str]] = {"left": set(), "right": set()}
    for side, value in (("left", left), ("right", right)):
        seen: set[str] = set()
        for kind, values in (
            ("criterion", value.criteria),
            ("attribute", value.attributes),
        ):
            for item in values:
                if item.id in seen:
                    raise ValueError(f"duplicate input ID on {side}: {item.id}")
                seen.add(item.id)
                reference = f"{side}:{kind}:{item.id}"
                items[reference] = item
                sides[side].add(reference)
    return items, sides["left"], sides["right"]


def validate_eval_report(
    report: EvalReport,
    left: CriteriaAttributeSet,
    right: CriteriaAttributeSet,
) -> None:
    _items, left_refs, right_refs = source_references(left, right)
    expected = left_refs | right_refs
    submitted = [
        reference
        for group in report.groups
        for reference in [*group.left_ids, *group.right_ids]
    ]
    counts = Counter(submitted)
    duplicates = sorted(reference for reference, count in counts.items() if count > 1)
    missing = sorted(expected - counts.keys())
    unexpected = sorted(counts.keys() - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            "Eval report must cover every source exactly once; "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    for group in report.groups:
        invalid_left = sorted(set(group.left_ids) - left_refs)
        invalid_right = sorted(set(group.right_ids) - right_refs)
        if invalid_left or invalid_right:
            raise ValueError(
                "Eval report uses source IDs on the wrong side; "
                f"invalid_left={invalid_left}, invalid_right={invalid_right}"
            )


@tool("submit_eval_report", args_schema=EvalReport)
def submit_eval_report(groups: list[EvalGroup]) -> dict[str, object]:
    """Submit the complete semantic-relation report without canonicalizing items."""
    return EvalReport(groups=groups).model_dump(mode="json")
