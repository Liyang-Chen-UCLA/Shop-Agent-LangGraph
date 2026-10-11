"""Deterministic, offline checkers, one per rubric.

Each checker takes one sample dict and a `CheckContext` and returns `UnitResult`s.
Sample shapes (all JSON-compatible):

research         {"item_id", "context", "result": ResearchResult, "reference": {relevance}}
aggregation      {"report": EvalReport, "actual": CriteriaAttributeSet, "input_item_ids"}

Text comparison against source material removes every whitespace character first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..agents.eval.jev import deterministic_score
from ..agents.eval.schemas import EvalReport
from ..agents.eval.tools import review_fields
from .rubric import GoldOverlay, RubricSet, UnitResult, load_rubrics

WHITESPACE = re.compile(r"[\s​-‍⁠﻿]+")


@dataclass
class CheckContext:
    overlay: GoldOverlay | None = None
    rubrics: RubricSet = field(default_factory=load_rubrics)


def squash(text: Any) -> str:
    """Remove all whitespace; the only normalization applied before source comparison."""
    return WHITESPACE.sub("", "" if text is None else str(text))


def in_source(needle: Any, source: Any) -> bool:
    needle = squash(needle)
    return bool(needle) and needle in squash(source)


def result(unit: str, ok: bool, reason: str = "") -> UnitResult:
    return UnitResult(unit=unit, status="pass" if ok else "fail", reason=reason)


def not_applicable(reason: str) -> list[UnitResult]:
    return [UnitResult(unit="-", status="na", reason=reason)]


# --- Research ------------------------------------------------------------------------

def rs_1(sample, ctx):
    evidence = sample["result"]["evidence"]
    return [result(f"e{i}", in_source(e["source_text"], sample["context"]), "source_text must be a substring of the product text")
            for i, e in enumerate(evidence)] or not_applicable("no evidence")


def rs_2(sample, ctx):
    expected = (sample.get("reference") or {}).get("relevance", "relevant")
    out = sample["result"]
    ok = out["relevance"] == expected and out["item_id"] == sample["item_id"]
    return [result("relevance", ok, f"relevance={out['relevance']!r}, item_id={out['item_id']!r}")]


def rs_3(sample, ctx):
    units = []
    for i, e in enumerate(sample["result"]["evidence"]):
        parts = {key: e.get(key) for key in ("value", "unit") if e.get(key)}
        if not parts:
            continue
        missing = [key for key, text in parts.items() if not in_source(text, e["source_text"])]
        units.append(result(f"e{i}", not missing, f"{missing} not found in source_text" if missing else ""))
    return units or not_applicable("no evidence with a value or unit")


# --- Aggregation ---------------------------------------------------------------------

def _agg(sample: dict, ctx: CheckContext):
    report = sample["report"]
    report = report if isinstance(report, EvalReport) else EvalReport.model_validate(report)
    gold_ref = {entry["item"]["id"]: ref for ref, entry in report.items.items() if ref.startswith("gold:")}
    groups = {ref: g for g in report.match for ref in g.gold_ids}
    return report, gold_ref, groups


def ag_1(sample, ctx):
    if ctx.overlay is None:
        return not_applicable("no gold overlay")
    _, gold_ref, groups = _agg(sample, ctx)
    units = []
    for i in ctx.overlay.core_ids:
        group = groups.get(gold_ref.get(i))
        if i not in gold_ref:
            units.append(UnitResult(unit=i, status="unresolved", reason="core id is not in gold"))
        else:
            units.append(result(i, group is not None and len(group.gold_ids) == len(group.actual_ids) == 1,
                                "core dimension is missing from Market" if group is None else
                                f"{len(group.gold_ids)} gold to {len(group.actual_ids)} actual"))
    return units


def ag_3(sample, ctx):
    report, gold_ref, groups = _agg(sample, ctx)
    units = []
    for item_id, ref in gold_ref.items():
        group = groups.get(ref)
        if group is None:
            continue
        gold = report.items[ref]
        actual = [report.items[a] for a in group.actual_ids]
        for name in (f for f in review_fields(ref, group, report.items) if f != "aliases"):
            unit = f"{item_id}.{name}"
            score = deterministic_score(name, gold, actual)
            if name in {"kind", "type"} and score is None:
                units.append(UnitResult(unit=unit, status="unresolved", reason="mixed group needs semantic judgement"))
            elif name == "kind":
                units.append(result(unit, score != 0, f"gold is {gold['kind']}, actual is {sorted({a['kind'] for a in actual})}"))
            elif name == "type":
                units.append(result(unit, score != 0, f"gold is {gold['item']['type']}, actual is {sorted({a['item']['type'] for a in actual})}"))
            else:
                side = "gold only" if name in gold["item"] else "actual only"
                units.append(result(unit, score != 0, f"field exists on {side}"))
    return units or not_applicable("no matched dimension")


def ag_6(sample, ctx):
    if ctx.overlay is None:
        return not_applicable("no gold overlay")
    actual = sample["actual"]
    dims = [d for key in ("criteria", "attributes") for d in actual[key]]
    keywords = [squash(k) for k in ctx.overlay.excluded_keywords]
    units = []
    for d in dims:
        hits = [k for k in keywords if any(k in squash(t) for t in [d["name"], *d.get("aliases", [])])]
        units.append(result(d["id"], not hits, f"excluded keywords: {hits}" if hits else ""))
    return units or not_applicable("no output dimensions")


def ag_7(sample, ctx):
    actual = sample["actual"]
    ids = [d["id"] for key in ("criteria", "attributes") for d in actual[key]]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    return [result("unique_ids", not duplicates, f"duplicate ids: {duplicates}" if duplicates else ""),
            result("source_item_ids", list(actual["source_item_ids"]) == list(sample["input_item_ids"]),
                   "source_item_ids must equal the input product IDs")]


CHECKERS: dict[str, Callable[[dict, CheckContext], list[UnitResult]]] = {
    "RS-1": rs_1, "RS-2": rs_2, "RS-3": rs_3,
    "AG-1": ag_1, "AG-3": ag_3, "AG-6": ag_6, "AG-7": ag_7,
}


def run_module(module: str, samples: dict[str, dict], *, context: CheckContext | None = None,
               only: set[str] | None = None) -> dict[str, list[UnitResult]]:
    """Run every rubric of a module over named samples.

    Units are prefixed `<sample>/<unit>` so several samples pool into one trial.
    """
    context = context or CheckContext()
    results: dict[str, list[UnitResult]] = {}
    for rubric in context.rubrics.module(module):
        if only is not None and rubric.id not in only:
            continue
        run = CHECKERS[rubric.id]
        units: list[UnitResult] = []
        for name, sample in samples.items():
            for unit in run(sample, context):
                units.append(unit.model_copy(update={"unit": f"{name}/{unit.unit}"}))
        results[rubric.id] = units
    return results
