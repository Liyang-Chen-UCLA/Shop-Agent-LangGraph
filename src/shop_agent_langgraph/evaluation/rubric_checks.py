"""Deterministic, offline checkers for every `judge: code` rubric.

Each checker takes one sample dict and a `CheckContext` and returns `UnitResult`s.
Sample shapes (all JSON-compatible):

e2e              {"input", "reference": {action, node_id, allowed_node_ids},
                  "trace": {"tool_calls": [{name, args, output, error}], "final_reply"}}
intent           {"input", "output": IntentResult, "reference": {action, category, ...}}
route            {"input", "output": RouteResult, "reference": {status, node_id, acceptable_node_ids}}
market_screening {"result": MarketResult, "reference": {irrelevant_item_ids, relevant_item_ids},
                  "page_texts": {item_id: {page_id: text}} (optional override)}
research         {"item_id", "context", "result": ResearchResult, "reference": {relevance}}
aggregation      {"report": EvalReport, "actual": CriteriaAttributeSet, "input_item_ids"}

Text comparison against source material removes every whitespace character first.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from ..agents.eval.jev import deterministic_score
from ..agents.eval.schemas import EvalReport
from ..agents.eval.tools import review_fields
from ..core.config import CONFIG
from ..domain.taxonomy import get_children, get_nodes
from .rubric import GoldOverlay, RubricSet, UnitResult, load_rubrics

MIN_SELECTED_PRODUCTS = 3
CHAIN_TOOLS = {"call_intent_agent": "intent", "call_route_agent": "route", "call_market_agent": "market"}
# Kept in sync with supervisor.tools.SUPERVISOR_TOOLS by a test; plus the route's taxonomy tools.
INTERNAL_TOOL_NAMES = ("call_intent_agent", "call_route_agent", "call_market_agent", "call_relation_agent",
                       "prepare_personalization", "complete_personalization", "taxonomy_search_nodes",
                       "taxonomy_get_nodes", "taxonomy_get_children", "submit_market_selection")
WHITESPACE = re.compile(r"[\s​-‍⁠﻿]+")


@dataclass
class CheckContext:
    overlay: GoldOverlay | None = None
    rubrics: RubricSet = field(default_factory=load_rubrics)
    max_selected: int = CONFIG.market.max_search_products


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


def parse(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except ValueError:
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def normalize_text(value: str) -> str:
    return squash(unicodedata.normalize("NFKC", value).casefold())


# --- E2E ----------------------------------------------------------------------------

def trace_from_messages(messages: list) -> dict:
    """Convert LangChain supervisor messages into the e2e trace shape."""
    outputs = {m.tool_call_id: m for m in messages if getattr(m, "type", None) == "tool"}
    calls = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            answer = outputs.get(call["id"])
            calls.append({"name": call["name"], "args": call.get("args", {}),
                          "output": None if answer is None else answer.content,
                          "error": answer is not None and getattr(answer, "status", None) == "error"})
    final = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai"
                  and not getattr(m, "tool_calls", None)), None)
    return {"tool_calls": calls, "final_reply": "" if final is None else str(final.content)}


def _calls(sample: dict, name: str) -> list[dict]:
    return [c for c in sample["trace"]["tool_calls"] if c["name"] == name]


def _route(sample: dict) -> dict | None:
    routes = [parse(c.get("output")) for c in _calls(sample, "call_route_agent")]
    routes = [r for r in routes if r]
    return routes[-1] if routes else None


def _resolved_id(route: dict | None) -> str | None:
    nodes = (route or {}).get("resolved_nodes") or []
    return str(nodes[0]["node_id"]) if route and route.get("status") == "resolved" and len(nodes) == 1 else None


def _action(sample: dict) -> str | None:
    action = (sample.get("reference") or {}).get("action")
    if action:
        return action
    intents = [parse(c.get("output")) for c in _calls(sample, "call_intent_agent")]
    intents = [i for i in intents if i]
    return intents[0].get("action") if intents else None


def e2e_1(sample, ctx):
    action = _action(sample)
    if action is None:
        return [UnitResult(unit="call_chain", status="unresolved", reason="no reference or observed action")]
    if action not in {"create", "switch", "query"}:
        return not_applicable(f"call-chain contract is defined for create/switch/query, not {action}")
    observed = [CHAIN_TOOLS[c["name"]] for c in sample["trace"]["tool_calls"] if c["name"] in CHAIN_TOOLS]
    if action == "query":
        expected = ["intent"]
    else:
        route = _route(sample)
        expected = ["intent", "route"] + (["market"] if _resolved_id(route) else [])
        if route is None and "route" in observed:
            return [result("call_chain", False, "route output could not be parsed")]
    return [result("call_chain", observed == expected, f"expected {expected}, observed {observed}")]


def e2e_2(sample, ctx):
    markets = _calls(sample, "call_market_agent")
    if not markets:
        return not_applicable("no market call")
    resolved = _resolved_id(_route(sample))
    return [result(f"market_call_{i}", resolved is not None and str(c.get("args", {}).get("node_id")) == resolved,
                   f"market node_id={c.get('args', {}).get('node_id')!r}, route resolved={resolved!r}")
            for i, c in enumerate(markets)]


def e2e_3(sample, ctx):
    reference = sample.get("reference") or {}
    if reference.get("route_status") == "ambiguous":
        return not_applicable("reference expects an ambiguous route, so no resolved node")
    allowed = {str(x) for x in reference.get("allowed_node_ids") or []}
    if reference.get("node_id") is not None:
        allowed.add(str(reference["node_id"]))
    if not allowed:
        return not_applicable("reference defines no expected node")
    resolved = _resolved_id(_route(sample))
    return [result("resolved_node", resolved in allowed, f"resolved={resolved!r}, allowed={sorted(allowed)}")]


def e2e_4(sample, ctx):
    markets = _calls(sample, "call_market_agent")
    if not markets:
        return not_applicable("no market call")
    units = []
    for i, call in enumerate(markets):
        output = parse(call.get("output"))
        status = None if output is None else output.get("status")
        units.append(result(f"market_call_{i}", not call.get("error") and status == "completed",
                            f"status={status!r}, error={bool(call.get('error'))}"))
    return units


JSON_PATTERNS = (re.compile(r'\{\s*"[^"\n]+"\s*:'), re.compile(r'"\w+"\s*:\s*(?:"|\[|\{|\d|true|false|null)'),
                 re.compile(r'\[\s*\{'))
AGENT_NAME = re.compile(r"(?i)\b(?:intent|route|market|research|relation|personalization|supervisor|eval)[\s_\-]*agent\b|\b\w+_agent\b")
INTERNAL_KEY = re.compile(r"\b(?:node_id|item_id|task_id|thread_id|source_item_ids?|resolved_nodes?)\b")
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
LONG_DIGITS = re.compile(r"(?<!\d)\d{9,}(?!\d)")


def leaks(reply: str, node_ids: list[str]) -> list[str]:
    found = []
    if any(p.search(reply) for p in JSON_PATTERNS):
        found.append("json")
    found += [f"tool:{name}" for name in INTERNAL_TOOL_NAMES if name in reply]
    found += [f"agent:{m.group(0)}" for m in AGENT_NAME.finditer(reply)]
    found += [f"id:{m.group(0)}" for pattern in (INTERNAL_KEY, UUID, LONG_DIGITS) for m in pattern.finditer(reply)]
    for node_id in node_ids:  # a bare number may be a budget; only labelled node IDs count
        if re.search(rf"(?:节点|node|ID|id|编号)\s*[:：#]?\s*{re.escape(node_id)}(?!\d)", reply):
            found.append(f"id:{node_id}")
    return found


def e2e_6(sample, ctx):
    reply = sample["trace"].get("final_reply") or ""
    if not reply.strip():
        return [UnitResult(unit="final_reply", status="unresolved", reason="no final reply")]
    route = _route(sample) or {}
    node_ids = [str(n["node_id"]) for key in ("resolved_nodes", "candidates", "children") for n in route.get(key) or []]
    found = leaks(reply, node_ids)
    return [result("final_reply", not found, f"leaked: {found}" if found else "")]


def e2e_7(sample, ctx):
    route = _route(sample)
    if not route or route.get("status") != "ambiguous":
        return not_applicable("route was not ambiguous")
    reply = sample["trace"].get("final_reply") or ""
    return [result(f"candidate:{c['node_id']}", in_source(c["node_name"], reply), f"candidate name {c['node_name']!r}")
            for c in route.get("candidates") or []] or [UnitResult(unit="candidates", status="fail", reason="ambiguous without candidates")]


# --- Intent -------------------------------------------------------------------------

def int_1(sample, ctx):
    reference = (sample.get("reference") or {}).get("action")
    if reference is None:
        return not_applicable("reference has no action")
    return [result("action", sample["output"]["action"] == reference, f"expected {reference!r}, got {sample['output']['action']!r}")]


def int_2(sample, ctx):
    reference = sample.get("reference") or {}
    if "category" not in reference:
        return not_applicable("reference has no category")
    output = sample["output"]["category"]
    if reference["category"] is None:
        return [result("category", output is None, f"expected null, got {output!r}")]
    ok = output is not None and normalize_text(output) == normalize_text(reference["category"])
    return [result("category", ok, f"expected {reference['category']!r}, got {output!r}")]


NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def int_5(sample, ctx):
    action = (sample.get("reference") or {}).get("action") or sample["output"]["action"]
    if action not in {"create", "update", "switch"}:
        return not_applicable(f"numbers must be preserved for create/update/switch, not {action}")
    numbers = list(dict.fromkeys(m.replace(",", "") for m in NUMBER.findall(unicodedata.normalize("NFKC", sample["input"]))))
    if not numbers:
        return not_applicable("no number in the user message")
    output = sample["output"]
    text = unicodedata.normalize("NFKC", "\n".join(
        [output.get("category") or "", *output["criteria_preferences"], *output["attribute_preferences"]])).replace(",", "")
    return [result(f"number:{n}", bool(re.search(rf"(?<![\d.]){re.escape(n)}(?!\d|\.\d)", text)),
                   f"{n} must appear verbatim in the category or preferences") for n in numbers]


# --- Route --------------------------------------------------------------------------

def _expected_nodes(reference: dict) -> set[str]:
    ids = {str(x) for x in reference.get("acceptable_node_ids") or []}
    if reference.get("node_id") is not None:
        ids.add(str(reference["node_id"]))
    return ids


def rt_1(sample, ctx):
    expected = (sample.get("reference") or {}).get("status")
    if expected is None:
        return not_applicable("reference has no status")
    return [result("status", sample["output"]["status"] == expected, f"expected {expected!r}, got {sample['output']['status']!r}")]


def rt_2(sample, ctx):
    reference = sample.get("reference") or {}
    if reference.get("status") != "resolved":
        return not_applicable("reference does not require a resolved node")
    output = sample["output"]
    if output["status"] != "resolved" or len(output["resolved_nodes"]) != 1:
        return [result("node_id", False, "output did not resolve exactly one node")]
    node_id = str(output["resolved_nodes"][0]["node_id"])
    return [result("node_id", node_id in _expected_nodes(reference), f"got {node_id!r}, expected {sorted(_expected_nodes(reference))}")]


def rt_3(sample, ctx):
    output = sample["output"]
    nodes = [(f"{key}:{n['node_id']}", n) for key in ("resolved_nodes", "candidates", "children") for n in output[key]]
    if not nodes:
        return not_applicable("no node returned")
    found, _ = get_nodes(list(dict.fromkeys(str(n["node_id"]) for _, n in nodes)))
    known = {n["node_id"]: n for n in found}
    units = []
    for unit, node in nodes:
        real = known.get(str(node["node_id"]))
        if real is None:
            units.append(result(unit, False, "node does not exist in taxonomy"))
        else:
            ok = real["node_name"] == node["node_name"] and real["node_path"] == node["node_path"]
            units.append(result(unit, ok, "" if ok else f"taxonomy has {real['node_name']!r} / {real['node_path']!r}"))
    return units


def rt_4(sample, ctx):
    output, reference = sample["output"], sample.get("reference") or {}
    if output["status"] != "ambiguous":
        return not_applicable("output was not ambiguous")
    expected = _expected_nodes(reference)
    if not expected:
        return not_applicable("reference defines no expected node")
    candidates = {str(n["node_id"]) for n in output["candidates"]}
    return [result("candidates", bool(candidates & expected), f"candidates={sorted(candidates)}, expected one of {sorted(expected)}")]


def rt_5(sample, ctx):
    output = sample["output"]
    if output["status"] != "resolved" or len(output["resolved_nodes"]) != 1:
        return not_applicable("no resolved node")
    node_id = str(output["resolved_nodes"][0]["node_id"])
    expected = {c["node_id"] for c in get_children([node_id])[0]["children"]}
    ids = [str(c["node_id"]) for c in output["children"]]
    ok = set(ids) == expected and len(ids) == len(set(ids))
    return [result("children", ok, f"missing={sorted(expected - set(ids))}, extra={sorted(set(ids) - expected)}, duplicates={len(ids) - len(set(ids))}")]


# --- Market screening ----------------------------------------------------------------

def _screening(sample: dict) -> tuple[dict[str, dict], list[str], list[str]]:
    entries = [e for e in sample["result"].get("audit", []) if e.get("stage") == "screening"]
    selected = [e["item_id"] for e in entries if e.get("selected")]
    return {e["item_id"]: e for e in entries}, selected, list(sample["result"].get("item_ids") or [])


def ms_1(sample, ctx):
    entries, selected, delivered = _screening(sample)
    ids = list(dict.fromkeys(selected + delivered))
    if not ids:
        return not_applicable("no product was selected")
    units = []
    for item_id in ids:
        problems = []
        entry = entries.get(item_id)
        if entry is None:
            problems.append("not among searched candidates")
        elif not (entry.get("read_page_ids") or entry.get("full_context_read")):
            problems.append("no page was read")
        if selected.count(item_id) > 1 or delivered.count(item_id) > 1:
            problems.append("duplicate")
        if item_id in delivered and item_id not in selected:
            problems.append("delivered without being selected")
        units.append(result(item_id, not problems, "; ".join(problems)))
    return units


def _page_text(sample: dict, item_id: str, page_id: str) -> str | None:
    override = (sample.get("page_texts") or {}).get(item_id)
    if override is not None:
        return override.get(page_id)
    from ..domain.market_env import get_product_pages
    try:
        pages = get_product_pages(item_id)
    except ValueError:
        return None
    return next((p["text"] for p in pages if p["page_id"] == page_id), None)


def ms_2(sample, ctx):
    entries, selected, _ = _screening(sample)
    units = []
    for item_id in dict.fromkeys(selected):
        entry = entries[item_id]
        evidence = entry.get("evidence") or []
        if not evidence:
            units.append(result(f"{item_id}#none", False, "selected product has no screening evidence"))
        for i, ev in enumerate(evidence):
            unit = f"{item_id}#{i}"
            if ev["page_id"] not in (entry.get("read_page_ids") or []):
                units.append(result(unit, False, f"page {ev['page_id']} was not read"))
                continue
            text = _page_text(sample, item_id, ev["page_id"])
            if text is None:
                units.append(UnitResult(unit=unit, status="unresolved", reason="page text unavailable"))
            else:
                units.append(result(unit, in_source(ev["source_text"], text), f"quote must be a substring of {ev['page_id']}"))
    return units or not_applicable("no product was selected")


def ms_3(sample, ctx):
    _, selected, _ = _screening(sample)
    irrelevant = (sample.get("reference") or {}).get("irrelevant_item_ids")
    if irrelevant is None or not selected:
        return not_applicable("no reference irrelevant list or no selection")
    return [result(i, i not in set(irrelevant), "reference marks this product irrelevant") for i in dict.fromkeys(selected)]


def ms_4(sample, ctx):
    count = len(set(_screening(sample)[1]))
    return [result("selected_count", MIN_SELECTED_PRODUCTS <= count <= ctx.max_selected,
                   f"{count} selected; allowed {MIN_SELECTED_PRODUCTS}..{ctx.max_selected}")]


def ms_5(sample, ctx):
    relevant = (sample.get("reference") or {}).get("relevant_item_ids")
    if not relevant:
        return not_applicable("reference has no relevant list")
    selected = set(_screening(sample)[1])
    return [result(i, i in selected, "reference-relevant product was not selected") for i in relevant]


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


def _core(sample, ctx):
    if ctx.overlay is None:
        return None, None
    report, gold_ref, groups = _agg(sample, ctx)
    return set(ctx.overlay.core_ids), (report, gold_ref, groups)


def ag_1(sample, ctx):
    core, parsed = _core(sample, ctx)
    if core is None:
        return not_applicable("no gold overlay")
    _, gold_ref, groups = parsed
    return [UnitResult(unit=i, status="unresolved", reason="core id is not in gold") if i not in gold_ref
            else result(i, gold_ref[i] in groups, "core dimension is missing from Market") for i in ctx.overlay.core_ids]


def ag_2(sample, ctx):
    core, parsed = _core(sample, ctx)
    if core is None:
        return not_applicable("no gold overlay")
    _, gold_ref, groups = parsed
    return [result(i, ref in groups, "dimension is missing from Market") for i, ref in gold_ref.items() if i not in core] \
        or not_applicable("no non-core gold dimensions")


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
            else:
                units.append(result(unit, score != 0, "kind/type disagree or the field exists on only one side"))
    return units or not_applicable("no matched dimension")


def ag_4(sample, ctx):
    core, parsed = _core(sample, ctx)
    if core is None:
        return not_applicable("no gold overlay")
    _, gold_ref, groups = parsed
    units = []
    for i in ctx.overlay.core_ids:
        group = groups.get(gold_ref.get(i))
        if i not in gold_ref:
            units.append(UnitResult(unit=i, status="unresolved", reason="core id is not in gold"))
        else:
            units.append(result(i, group is not None and len(group.gold_ids) == len(group.actual_ids) == 1,
                                "core dimension is missing" if group is None else
                                f"{len(group.gold_ids)} gold to {len(group.actual_ids)} actual"))
    return units


def ag_5(sample, ctx):
    """LLM rubric whose judgement already exists: read the EvalReport's 0/1 field scores."""
    report, gold_ref, _ = _agg(sample, ctx)
    fields = {f.name for f in ctx.rubrics.by_id["AG-5"].fields if not f.auxiliary}
    ids = {ref: i for i, ref in gold_ref.items()}
    return [result(f"{ids[review.gold_ref]}.{score.field}", score.score == 1, score.reason)
            for review in report.reviews for score in review.fields if score.field in fields] \
        or not_applicable("no matched dimension has an applicable field")


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
    "E2E-1": e2e_1, "E2E-2": e2e_2, "E2E-3": e2e_3, "E2E-4": e2e_4, "E2E-6": e2e_6, "E2E-7": e2e_7,
    "INT-1": int_1, "INT-2": int_2, "INT-5": int_5,
    "RT-1": rt_1, "RT-2": rt_2, "RT-3": rt_3, "RT-4": rt_4, "RT-5": rt_5,
    "MS-1": ms_1, "MS-2": ms_2, "MS-3": ms_3, "MS-4": ms_4, "MS-5": ms_5,
    "RS-1": rs_1, "RS-2": rs_2, "RS-3": rs_3,
    "AG-1": ag_1, "AG-2": ag_2, "AG-3": ag_3, "AG-4": ag_4, "AG-6": ag_6, "AG-7": ag_7,
}
# Existing judgments that need no new model call.
BUILTIN_JUDGES: dict[str, Callable[[dict, CheckContext], list[UnitResult]]] = {"AG-5": ag_5}
Judge = Callable[[dict, CheckContext], list[UnitResult]]


def run_module(module: str, samples: dict[str, dict], *, context: CheckContext | None = None,
               judges: dict[str, Judge] | None = None, only: set[str] | None = None) -> dict[str, list[UnitResult]]:
    """Run every applicable rubric of a module over named samples.

    Units are prefixed `<sample>/<unit>` so several samples pool into one trial.
    llm rubrics run only through an injected judge (or a built-in over existing data);
    without one they are simply absent from the result.
    """
    context = context or CheckContext()
    judges = {**BUILTIN_JUDGES, **(judges or {})}
    results: dict[str, list[UnitResult]] = {}
    for rubric in context.rubrics.module(module):
        if only is not None and rubric.id not in only:
            continue
        run = CHECKERS.get(rubric.id) if rubric.judge == "code" else judges.get(rubric.id)
        if run is None:
            continue
        units: list[UnitResult] = []
        for name, sample in samples.items():
            for unit in run(sample, context):
                units.append(unit.model_copy(update={"unit": f"{name}/{unit.unit}"}))
        results[rubric.id] = units
    return results
