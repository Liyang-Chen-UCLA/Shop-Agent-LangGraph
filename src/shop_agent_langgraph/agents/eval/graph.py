from __future__ import annotations
import asyncio
import json
from pathlib import Path
from threading import Lock
from copy import deepcopy
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from ...core.llm import build_deepseek_model
from ...domain.criteria import CriteriaAttributeSet
from .schemas import Coverage, EvalMetrics, EvalReport, FieldScore, MatchingReport, SchemaReview
from .tools import review_fields, source_references, validate_eval_report, validate_schema_review, validate_scope

PROMPT_PATH = Path(__file__).with_name("prompt.md")


class EvalAgent:
    """Read-only two-stage gold comparison with deterministic binary bookkeeping."""

    def __init__(self, model: BaseChatModel, *, jev=None, config: dict | None = None,
                 use_deterministic_fields: bool = False, review_batch_size: int | None = None):
        self.jev = jev
        self.config = config or {}
        self.use_deterministic_fields = use_deterministic_fields
        if review_batch_size is not None and review_batch_size < 1:
            raise ValueError("review batch size must be positive")
        self.review_batch_size = review_batch_size
        self.match_graph = create_agent(model=model, tools=[],
            system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
            response_format=ToolStrategy(MatchingReport), name="eval_matching_agent")
        self.review_graph = create_agent(model=model, tools=[],
            system_prompt=PROMPT_PATH.with_name("schema_prompt.md").read_text(encoding="utf-8"),
            response_format=ToolStrategy(SchemaReview), name="eval_schema_agent")

    @staticmethod
    async def _judge(graph, schema, payload, validate, *, sync=False, config=None):
        messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        for attempt in range(2):
            if sync:
                state = graph.invoke({"messages": messages}, config={**(config or {}), "recursion_limit": 6})
            else:
                state = await graph.ainvoke({"messages": messages}, config={**(config or {}), "recursion_limit": 6})
            try:
                raw = state["structured_response"]
                result = raw if isinstance(raw, schema) else schema.model_validate(raw)
                validate(result)
                return result
            except ValueError as error:
                if attempt:
                    raise
                messages = list(state["messages"])
                messages.append({"role": "user", "content": f"Previous report rejected: {error}. Return a complete corrected report."})
        raise AssertionError("unreachable")

    async def ainvoke(self, gold: CriteriaAttributeSet, actual: CriteriaAttributeSet) -> EvalReport:
        return await self._evaluate(gold, actual)

    async def _evaluate(self, gold: CriteriaAttributeSet, actual: CriteriaAttributeSet, *, sync=False) -> EvalReport:
        validate_scope(gold, actual)
        items, gold_refs, actual_refs = source_references(gold, actual)
        if items:
            payload = {ref: {k: entry["item"][k] for k in ("name", "description")}
                       for ref, entry in items.items()}
            matching = await self._judge(self.match_graph, MatchingReport, {"items": payload},
                lambda report: validate_eval_report(report, gold, actual), sync=sync, config=self.config)
        else:
            matching = MatchingReport(match=[], missing=[], extra=[])
        if matching.match:
            groups = [{"gold": {r: items[r] for r in g.gold_ids},
                       "actual": {r: items[r] for r in g.actual_ids},
                       "required_fields": {r: review_fields(r, g, items) for r in g.gold_ids}}
                      for g in matching.match]
            known = {}
            payload_groups = deepcopy(groups)
            if self.use_deterministic_fields:
                from .jev import deterministic_score
                for group in payload_groups:
                    for ref, fields in group["required_fields"].items():
                        known[ref] = {field: score for field in fields
                            if (score := deterministic_score(field, group["gold"][ref], list(group["actual"].values()))) is not None}
                        group["required_fields"][ref] = [f for f in fields if f not in known[ref]]

            def validate_review(result, matching_scope=matching):
                if known:
                    expected_refs = {r for g in matching_scope.match for r in g.gold_ids}
                    if {r.gold_ref for r in result.items} != expected_refs or len(result.items) != len(expected_refs):
                        raise ValueError(f"review must cover these gold_ref keys exactly once: {sorted(expected_refs)}; received: {[r.gold_ref for r in result.items]}")
                    for item in result.items:
                        expected_fields = set(review_fields(item.gold_ref,
                            next(g for g in matching_scope.match if item.gold_ref in g.gold_ids), items)) - set(known[item.gold_ref])
                        if {s.field for s in item.fields} != expected_fields or len(item.fields) != len(expected_fields):
                            raise ValueError(f"Return ALL gold refs {sorted(expected_refs)}, not just the corrected item. "
                                f"For {item.gold_ref} return only these fields: {sorted(expected_fields)}. "
                                "For every other gold ref use its required_fields from the original payload.")
                        item.fields.extend(FieldScore(field=f, score=s,
                            reason="Determined from supplied classification/type or field presence") for f, s in known[item.gold_ref].items())
                validate_schema_review(result, matching_scope, items)

            if self.review_batch_size is None:
                review = await self._judge(self.review_graph, SchemaReview, {"groups": payload_groups},
                    validate_review, sync=sync, config=self.config)
            else:
                refs = [r for g in matching.match for r in g.gold_ids]
                semaphore = asyncio.Semaphore(3)

                async def batch(selected):
                    payload = [{**g, "gold": {r: e for r, e in g["gold"].items() if r in selected},
                        "required_fields": {r: f for r, f in g["required_fields"].items() if r in selected}}
                        for g in payload_groups if set(g["gold"]) & selected]
                    scope = MatchingReport(match=[g.model_copy(update={"gold_ids": [r for r in g.gold_ids if r in selected]})
                        for g in matching.match if set(g.gold_ids) & selected], missing=[], extra=[])
                    async with semaphore:
                        return await self._judge(self.review_graph, SchemaReview, {"groups": payload},
                            lambda result: validate_review(result, scope), sync=sync, config=self.config)

                batches = await asyncio.gather(*(batch(set(refs[i:i + self.review_batch_size]))
                    for i in range(0, len(refs), self.review_batch_size)))
                review = SchemaReview(items=[item for result in batches for item in result.items])
                validate_schema_review(review, matching, items)
            judge_details = await self.jev.review(groups, review) if self.jev is not None else {}
        else:
            review = SchemaReview(items=[])
            judge_details = {}
        by_ref = {r.gold_ref: r for r in review.items}
        ordered_reviews = []
        for group in matching.match:
            one_to_one = len(group.gold_ids) == len(group.actual_ids) == 1
            reason = ("One gold item matches one actual item" if one_to_one else
                      f"Merged/split items: {len(group.gold_ids)} gold to {len(group.actual_ids)} actual")
            for ref in group.gold_ids:
                item_review = by_ref[ref].model_copy(deep=True)
                item_review.fields.append(FieldScore(field="granularity", reason=reason, score=int(one_to_one)))
                ordered_reviews.append(item_review)
        matched_gold = {r for g in matching.match for r in g.gold_ids}
        matched_actual = {r for g in matching.match for r in g.actual_ids}

        def coverage(gold_ids, actual_ids):
            return Coverage(gold_count=len(gold_ids), actual_count=len(actual_ids),
                matched_gold_count=len(gold_ids & matched_gold),
                matched_actual_count=len(actual_ids & matched_actual),
                missing_count=len(gold_ids - matched_gold), extra_count=len(actual_ids - matched_actual))

        scores = [s for r in ordered_reviews for s in r.fields]
        field_scores = {field: (sum(s.score for s in scores if s.field == field) /
            sum(s.field == field for s in scores)) if any(s.field == field for s in scores) else None
            for field in ("kind", "type", "units", "formula", "direction", "values", "value_domain", "granularity")}
        metrics = EvalMetrics(**coverage(gold_refs, actual_refs).model_dump(),
            match_group_count=len(matching.match),
            classification_error_count=sum(s.field == "kind" and s.score == 0 for s in scores),
            granularity_error_count=sum(s.field == "granularity" and s.score == 0 for s in scores),
            field_scores=field_scores, by_kind={kind: coverage(
                {r for r in gold_refs if items[r]["kind"] == kind},
                {r for r in actual_refs if items[r]["kind"] == kind}) for kind in ("criterion", "attribute")})
        return EvalReport(**matching.model_dump(), source_item_ids=list(gold.source_item_ids),
                          items=items, reviews=ordered_reviews, metrics=metrics, judge_details=judge_details)

    def invoke(self, gold: CriteriaAttributeSet, actual: CriteriaAttributeSet) -> EvalReport:
        return asyncio.run(self._evaluate(gold, actual, sync=True))


def build_eval_agent(model: BaseChatModel | None = None) -> EvalAgent:
    return EvalAgent(model or build_deepseek_model())


class LazyEvalAgent:
    def __init__(self):
        self._instance: EvalAgent | None = None
        self._lock = Lock()

    def _get(self):
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = build_eval_agent()
        return self._instance

    def invoke(self, gold: CriteriaAttributeSet, actual: CriteriaAttributeSet) -> EvalReport:
        return self._get().invoke(gold, actual)

    async def ainvoke(self, gold: CriteriaAttributeSet, actual: CriteriaAttributeSet) -> EvalReport:
        return await self._get().ainvoke(gold, actual)


eval_agent = LazyEvalAgent()
