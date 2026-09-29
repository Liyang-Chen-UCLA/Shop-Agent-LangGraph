"""Evaluate normalized product facts without treating missing data as a match."""
from .schemas import Constraint


def evaluate_constraint(constraint: Constraint, value, unit=None) -> str:
    if value is None or unit != constraint.unit:
        return "unknown"
    expected = constraint.values[0]
    if type(value) is not type(expected) and not (
        type(value) in (int, float) and type(expected) in (int, float)
    ):
        return "unknown"
    if constraint.operator == "in":
        matches = value in constraint.values
    elif constraint.operator == "eq":
        matches = value == expected
    elif constraint.operator == "lte":
        matches = value <= expected
    else:
        matches = value >= expected
    return "match" if matches else "no_match"
