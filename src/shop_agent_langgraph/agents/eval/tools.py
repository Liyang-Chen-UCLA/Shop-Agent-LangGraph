from __future__ import annotations
from collections import Counter
from ...domain.criteria import CriteriaAttributeSet
from .schemas import MatchingReport, SchemaReview

SCHEMA_FIELDS = {"type", "units", "formula", "direction", "values", "value_domain"}


def source_references(gold: CriteriaAttributeSet, actual: CriteriaAttributeSet):
    """Do not leak kind or canonical IDs into name/description matching."""
    items: dict[str, dict] = {}
    sides: dict[str, set[str]] = {"gold": set(), "actual": set()}
    for side, collection in (("gold", gold), ("actual", actual)):
        if len(collection.source_item_ids) != len(set(collection.source_item_ids)):
            raise ValueError(f"duplicate product IDs on {side}")
        seen: set[str] = set()
        for kind, values in (("criterion", collection.criteria), ("attribute", collection.attributes)):
            for item in values:
                if item.id in seen:
                    raise ValueError(f"duplicate input ID on {side}: {item.id}")
                seen.add(item.id)
                ref = f"{side}:{len(sides[side])}"
                items[ref] = {"kind": kind, "item": item.model_dump(mode="json")}
                sides[side].add(ref)
    return items, sides["gold"], sides["actual"]


def validate_scope(gold: CriteriaAttributeSet, actual: CriteriaAttributeSet):
    if set(gold.source_item_ids) != set(actual.source_item_ids):
        raise ValueError("gold and actual must use the same fixed product item IDs")


def validate_eval_report(report: MatchingReport, gold: CriteriaAttributeSet, actual: CriteriaAttributeSet):
    _, gold_refs, actual_refs = source_references(gold, actual)
    submitted = [ref for g in report.match for ref in g.gold_ids + g.actual_ids]
    submitted += [i.ref for i in report.missing + report.extra]
    counts = Counter(submitted)
    expected = gold_refs | actual_refs
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise ValueError("matching must cover every source exactly once without unknown IDs")
    for group in report.match:
        if not set(group.gold_ids) <= gold_refs or not set(group.actual_ids) <= actual_refs:
            raise ValueError("match references are on the wrong side")
    if any(i.ref not in gold_refs for i in report.missing) or any(i.ref not in actual_refs for i in report.extra):
        raise ValueError("missing must reference gold; extra must reference actual")
    reasons = [g.reason for g in report.match] + [i.reason for i in report.missing + report.extra]
    if any(not reason.strip() for reason in reasons):
        raise ValueError("matching reasons must be nonblank")


def review_fields(gold_ref: str, group, items: dict) -> list[str]:
    field_names = set(items[gold_ref]["item"])
    if len(group.actual_ids) == 1:
        field_names.update(items[group.actual_ids[0]]["item"])
    return ["kind", *sorted(field_names & SCHEMA_FIELDS), "aliases"]


def validate_schema_review(review: SchemaReview, matching: MatchingReport, items: dict):
    groups = {ref: g for g in matching.match for ref in g.gold_ids}
    refs = [item.gold_ref for item in review.items]
    if set(refs) != set(groups) or len(refs) != len(set(refs)):
        raise ValueError("schema review must cover each matched gold item exactly once")
    for item in review.items:
        fields = [s.field for s in item.fields]
        expected = set(review_fields(item.gold_ref, groups[item.gold_ref], items))
        if set(fields) != expected or len(fields) != len(set(fields)):
            raise ValueError(f"schema review requires exactly these fields for {item.gold_ref}: {sorted(expected)}")
        if any(not s.reason.strip() for s in item.fields):
            raise ValueError("field reasons must be nonblank")
        gold_item = items[item.gold_ref]["item"]
        actual_items = [items[ref]["item"] for ref in groups[item.gold_ref].actual_ids]
        for score in item.fields:
            if score.field in SCHEMA_FIELDS:
                gold_has = score.field in gold_item
                actual_has = any(score.field in actual for actual in actual_items)
                if gold_has != actual_has and score.score != 0:
                    raise ValueError(f"{score.field} is absent on one side and must score 0")
        actual_types = {actual["type"] for actual in actual_items}
        if len(actual_types) == 1:
            if next(s.score for s in item.fields if s.field == "type") != int(gold_item["type"] in actual_types):
                raise ValueError("type score contradicts the supplied schema types")
        actual_kinds = {items[ref]["kind"] for ref in groups[item.gold_ref].actual_ids}
        if len(actual_kinds) == 1:
            expected_score = int(items[item.gold_ref]["kind"] in actual_kinds)
            if next(s.score for s in item.fields if s.field == "kind") != expected_score:
                raise ValueError("kind score contradicts the supplied criterion/attribute classification")
