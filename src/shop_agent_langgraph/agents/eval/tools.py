from __future__ import annotations

from typing import Any

from langchain.tools import tool

from ...domain.criteria import Attribute, CriteriaAttributeSet, Criterion
from .schemas import (
    CanonicalOutput,
    FinalizeRequest,
    IndependentBatch,
    MatchBatch,
    MatchDecision,
    PartialResolution,
    PartialResolutionBatch,
)


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
    for side, value in (("left", left), ("right", right)):
        seen: set[str] = set()
        for kind, items in (("criterion", value.criteria), ("attribute", value.attributes)):
            for item in items:
                if item.id in seen:
                    raise ValueError(f"duplicate input ID on {side}: {item.id}")
                seen.add(item.id)
                reference = f"{side}:{kind}:{item.id}"
                unresolved[reference] = item
                kinds[reference] = kind

    criteria: list[Criterion] = []
    attributes: list[Attribute] = []
    source_item_ids = list(dict.fromkeys([*left.source_item_ids, *right.source_item_ids]))

    def validate_refs(refs: list[str], side: str | None = None) -> None:
        if len(refs) != len(set(refs)):
            raise ValueError("source references must not be reused within a batch/group")
        for ref in refs:
            if ref not in unresolved:
                raise ValueError(f"unknown or already consumed source reference: {ref}")
            if side and not ref.startswith(f"{side}:"):
                raise ValueError(f"expected a {side} source reference: {ref}")

    def commit(refs: list[str], outputs: list[CanonicalOutput]) -> None:
        # Preflight the entire batch before any mutation, including output validation.
        validate_refs(refs)
        ids = [item.id for item in [*criteria, *attributes]]
        ids.extend(output.item.id for output in outputs)
        if len(ids) != len(set(ids)):
            raise ValueError("canonical output IDs must be unique across criteria and attributes")
        new_criteria = [output.item for output in outputs if output.kind == "criterion"]
        new_attributes = [output.item for output in outputs if output.kind == "attribute"]
        validated = CriteriaAttributeSet(
            source_item_ids=source_item_ids,
            criteria=[*criteria, *new_criteria],
            attributes=[*attributes, *new_attributes],
        )
        for ref in refs:
            del unresolved[ref]
        criteria[:] = validated.criteria
        attributes[:] = validated.attributes

    @tool("match", args_schema=MatchBatch)
    def match(matches: list[MatchDecision]) -> dict[str, Any]:
        """Atomically merge a batch of semantically equivalent left/right pairs."""
        decisions = [MatchDecision.model_validate(value) for value in matches]
        refs: list[str] = []
        for decision in decisions:
            validate_refs([decision.left_id], "left")
            validate_refs([decision.right_id], "right")
            refs.extend([decision.left_id, decision.right_id])
        commit(refs, decisions)
        return {"committed": True, "accepted": len(decisions), "unresolved": list(unresolved)}

    @tool("independent", args_schema=IndependentBatch)
    def independent(item_ids: list[str]) -> dict[str, Any]:
        """Atomically retain a batch of distinct unresolved concepts without rewriting them."""
        validate_refs(item_ids)
        outputs = [
            CanonicalOutput(kind=kinds[ref], item=unresolved[ref]) for ref in item_ids
        ]
        commit(item_ids, outputs)
        return {"committed": True, "accepted": len(item_ids), "unresolved": list(unresolved)}

    @tool("resolve_partial", args_schema=PartialResolutionBatch)
    def resolve_partial(resolutions: list[PartialResolution]) -> dict[str, Any]:
        """Resolve a batch of many-to-many semantic overlaps into shared and distinct outputs.

        Each group accepts multiple left_ids and right_ids and produces multiple outputs.
        Preserve every source's meaning through source_refs and preserved_aspects.
        All groups are validated before any writes. Invalid batches change no state.
        needs_review groups retain their inputs and block final submission.
        """
        try:
            groups = [PartialResolution.model_validate(value) for value in resolutions]
            if len({group.group_id for group in groups}) != len(groups):
                raise ValueError("group_id must be unique within the batch")
            all_refs: list[str] = []
            consumed: list[str] = []
            outputs: list[CanonicalOutput] = []
            results: list[dict[str, Any]] = []
            for group in groups:
                validate_refs(group.left_ids, "left")
                validate_refs(group.right_ids, "right")
                group_refs = set(group.left_ids + group.right_ids)
                all_refs.extend(group.left_ids + group.right_ids)
                covered: set[str] = set()
                for output in group.outputs:
                    refs = set(output.source_refs)
                    if len(refs) != len(output.source_refs) or not refs <= group_refs:
                        raise ValueError(f"{group.group_id}: invalid output source_refs")
                    both_sides = bool(refs & set(group.left_ids)) and bool(
                        refs & set(group.right_ids)
                    )
                    if (output.alignment == "match") != both_sides:
                        raise ValueError(
                            f"{group.group_id}: match outputs require both sides; "
                            "independent outputs must reference one side only"
                        )
                    if any(not aspect.strip() for aspect in output.preserved_aspects):
                        raise ValueError(f"{group.group_id}: preserved_aspects must be nonblank")
                    covered.update(refs)
                if group.status == "resolved":
                    if covered != group_refs:
                        raise ValueError(f"{group.group_id}: every source must map to an output")
                    consumed.extend(group.left_ids + group.right_ids)
                    outputs.extend(group.outputs)
                results.append(
                    {
                        "group_id": group.group_id,
                        "relation": group.relation,
                        "reason": group.reason,
                        "status": group.status,
                        "consumed_refs": (
                            group.left_ids + group.right_ids
                            if group.status == "resolved"
                            else []
                        ),
                        "outputs": [
                            output.model_dump(mode="json")
                            for output in group.outputs
                        ],
                        "missing_evidence": group.missing_evidence,
                    }
                )
            # A source can support several outputs inside one group, but not two groups.
            validate_refs(all_refs)
            commit(consumed, outputs)
        except ValueError as exc:
            return {"committed": False, "errors": [str(exc)], "unresolved": list(unresolved)}
        return {"committed": True, "results": results, "unresolved": list(unresolved)}

    @tool("submit", args_schema=FinalizeRequest)
    def submit() -> dict[str, Any]:
        """Finalize from accepted operations only. Call alone, with no arguments."""
        if unresolved:
            raise ValueError(f"unresolved item IDs remain: {list(unresolved)}")
        return CriteriaAttributeSet(
            source_item_ids=source_item_ids, criteria=criteria, attributes=attributes,
        ).model_dump(mode="json")

    return [match, resolve_partial, independent, submit], unresolved, criteria, attributes
