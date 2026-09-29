from __future__ import annotations

import operator
from typing import Literal

from ...domain.criteria import SchemaModel
from .schemas import Claim, NodeRef, RelationGraph, Scalar


class ObservedValue(SchemaModel):
    value: Scalar
    unit: str | None = None


_OPERATORS = {"eq": operator.eq, "ne": operator.ne, "gt": operator.gt,
              "gte": operator.ge, "lt": operator.lt, "lte": operator.le}


def match_conditions(claim: Claim, values: dict[str, ObservedValue]) -> Literal["matched", "unmatched", "unknown"]:
    """Match AND conditions without inferring missing values or converting units."""
    unknown = False
    for condition in claim.conditions:
        left = values.get(condition.ref)
        right = (values.get(condition.value_ref) if condition.value_ref else
                 ObservedValue(value=condition.value, unit=condition.unit))
        if left is None or right is None or left.unit != right.unit:
            unknown = True
            continue
        if type(left.value) is not type(right.value):
            unknown = True
            continue
        if condition.operator not in ("eq", "ne") and (isinstance(left.value, bool) or not isinstance(left.value, (float, int))):
            unknown = True
            continue
        if not _OPERATORS[condition.operator](left.value, right.value):
            return "unmatched"
    return "unknown" if unknown else "matched"


def supported_paths(graph: RelationGraph, source: NodeRef, target: NodeRef,
                    values: dict[str, ObservedValue] | None = None, *, max_depth: int = 4,
                    max_paths: int = 20) -> list[list[str]]:
    """Return bounded, condition-matched relation-ID paths, not causal conclusions.

    No edge is reversed, no effect signs are multiplied, and no score is computed.
    Consumers must inspect each edge's kind, scope and statement before explaining it.
    """
    if not 1 <= max_depth <= 10 or not 1 <= max_paths <= 100:
        raise ValueError("max_depth must be 1..10 and max_paths 1..100")
    edges = [edge for edge in graph.relations if edge.status == "supported"
             and match_conditions(edge, values or {}) == "matched"]
    paths = []

    def walk(ref, visited, path):
        if len(paths) >= max_paths or len(path) >= max_depth:
            return
        for edge in edges:
            if len(paths) >= max_paths:
                break
            if edge.source != ref or edge.target in visited:
                continue
            extended = path + [edge.id]
            if edge.target == target:
                paths.append(extended)
            else:
                walk(edge.target, visited | {edge.target}, extended)

    walk(source, {source}, [])
    return paths
