from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph

from ...core.config import CONFIG
from ...domain.criteria import Attribute, CriteriaAttributeSet, Criterion
from ..eval.schemas import EvalReport
from ..eval.tools import validate_eval_report
from .schemas import (
    AggregationPlan,
    CanonicalOutput,
    MarketAggregationOutcome,
    MatchPatch,
    PartialResolution,
    PendingAggregation,
)


PROMPT_PATH = Path(__file__).with_name("aggregation_prompt.md")


@dataclass
class AggregationRuntime:
    left: CriteriaAttributeSet
    right: CriteriaAttributeSet
    unresolved: dict[str, Criterion | Attribute] = field(init=False, default_factory=dict)
    kinds: dict[str, str] = field(init=False, default_factory=dict)
    criteria: list[Criterion] = field(init=False, default_factory=list)
    attributes: list[Attribute] = field(init=False, default_factory=list)
    audit: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        for side, value in (("left", self.left), ("right", self.right)):
            seen: set[str] = set()
            for kind, items in (
                ("criterion", value.criteria),
                ("attribute", value.attributes),
            ):
                for item in items:
                    if item.id in seen:
                        raise ValueError(f"duplicate input ID on {side}: {item.id}")
                    seen.add(item.id)
                    reference = f"{side}:{kind}:{item.id}"
                    self.unresolved[reference] = item
                    self.kinds[reference] = kind

    @property
    def source_item_ids(self) -> list[str]:
        return list(
            dict.fromkeys(
                [*self.left.source_item_ids, *self.right.source_item_ids]
            )
        )

    def collection(self) -> CriteriaAttributeSet:
        if self.unresolved:
            raise ValueError(f"unresolved item IDs remain: {list(self.unresolved)}")
        return CriteriaAttributeSet(
            source_item_ids=self.source_item_ids,
            criteria=self.criteria,
            attributes=self.attributes,
        )

    def validate_refs(self, refs: list[str], side: str | None = None) -> None:
        if len(refs) != len(set(refs)):
            raise ValueError("source references must be unique")
        for ref in refs:
            if ref not in self.unresolved:
                raise ValueError(f"unknown or already consumed source reference: {ref}")
            if side and not ref.startswith(f"{side}:"):
                raise ValueError(f"expected a {side} source reference: {ref}")

    def validate_output_ids(self, outputs: list[CanonicalOutput]) -> None:
        ids = [item.id for item in [*self.criteria, *self.attributes]]
        ids.extend(output.item.id for output in outputs)
        if len(ids) != len(set(ids)):
            raise ValueError("canonical output IDs must be unique across criteria and attributes")

    def commit(
        self,
        refs: list[str],
        outputs: list[CanonicalOutput],
        audit_entries: list[dict[str, Any]],
    ) -> None:
        self.validate_refs(refs)
        self.validate_output_ids(outputs)
        new_criteria = [output.item for output in outputs if output.kind == "criterion"]
        new_attributes = [output.item for output in outputs if output.kind == "attribute"]
        validated = CriteriaAttributeSet(
            source_item_ids=self.source_item_ids,
            criteria=[*self.criteria, *new_criteria],
            attributes=[*self.attributes, *new_attributes],
        )
        for ref in refs:
            del self.unresolved[ref]
        self.criteria[:] = validated.criteria
        self.attributes[:] = validated.attributes
        self.audit.extend(audit_entries)

    @staticmethod
    def items_are_directly_compatible(
        left: Criterion | Attribute,
        right: Criterion | Attribute,
    ) -> bool:
        if type(left) is not type(right):
            return False
        ignored = {"id", "name", "description", "aliases"}
        left_payload = left.model_dump(mode="json", exclude=ignored)
        right_payload = right.model_dump(mode="json", exclude=ignored)
        return left_payload == right_payload

    def direct_match_output(self, left_ref: str, right_ref: str) -> CanonicalOutput:
        left_item = self.unresolved[left_ref]
        right_item = self.unresolved[right_ref]
        aliases = list(left_item.aliases)
        if right_item.name != left_item.name and right_item.name not in aliases:
            aliases.append(right_item.name)
        for alias in right_item.aliases:
            if alias != left_item.name and alias not in aliases:
                aliases.append(alias)
        merged = left_item.model_copy(deep=True, update={"aliases": aliases})
        return CanonicalOutput(kind=self.kinds[left_ref], item=merged)

    def apply_direct_groups(self, report: EvalReport) -> list[dict[str, Any]]:
        candidates: list[tuple[int, list[str], list[CanonicalOutput], dict[str, Any]]] = []
        hard_indices: set[int] = set()
        for index, group in enumerate(report.groups):
            group_id = f"group_{index}"
            refs = [*group.left_ids, *group.right_ids]
            if group.status == "independent":
                outputs = [
                    CanonicalOutput(kind=self.kinds[ref], item=self.unresolved[ref])
                    for ref in refs
                ]
                candidates.append(
                    (
                        index,
                        refs,
                        outputs,
                        {
                            "operation": "independent",
                            "group_id": group_id,
                            "source_refs": refs,
                            "outputs": [
                                output.item.model_dump(mode="json") for output in outputs
                            ],
                        },
                    )
                )
                continue
            if group.status == "match" and len(group.left_ids) == len(group.right_ids) == 1:
                left_ref = group.left_ids[0]
                right_ref = group.right_ids[0]
                if self.items_are_directly_compatible(
                    self.unresolved[left_ref], self.unresolved[right_ref]
                ):
                    output = self.direct_match_output(left_ref, right_ref)
                    candidates.append(
                        (
                            index,
                            refs,
                            [output],
                            {
                                "operation": "match",
                                "group_id": group_id,
                                "source_refs": refs,
                                "base_ref": left_ref,
                                "patch": {"aliases": output.item.aliases},
                                "outputs": [output.item.model_dump(mode="json")],
                            },
                        )
                    )
                    continue
            hard_indices.add(index)

        output_owners: dict[str, list[int]] = {}
        for index, _refs, outputs, _audit in candidates:
            for output in outputs:
                output_owners.setdefault(output.item.id, []).append(index)
        for owners in output_owners.values():
            if len(owners) > 1:
                hard_indices.update(owners)

        direct_refs: list[str] = []
        direct_outputs: list[CanonicalOutput] = []
        direct_audit: list[dict[str, Any]] = []
        for index, refs, outputs, audit_entry in candidates:
            if index in hard_indices:
                continue
            direct_refs.extend(refs)
            direct_outputs.extend(outputs)
            direct_audit.append(audit_entry)
        if direct_refs:
            self.commit(direct_refs, direct_outputs, direct_audit)

        return [
            {"group_id": f"group_{index}", **group.model_dump(mode="json")}
            for index, group in enumerate(report.groups)
            if index in hard_indices
        ]

    def patched_match_output(self, decision: MatchPatch) -> CanonicalOutput:
        self.validate_refs([decision.left_id], "left")
        self.validate_refs([decision.right_id], "right")
        if decision.base_ref not in {decision.left_id, decision.right_id}:
            raise ValueError("base_ref must be left_id or right_id")
        base = self.unresolved[decision.base_ref]
        payload = base.model_dump(mode="python")
        payload.update(decision.patch)
        merged = type(base).model_validate(payload)
        return CanonicalOutput(kind=self.kinds[decision.base_ref], item=merged)

    def apply_plan(
        self,
        plan: AggregationPlan,
        hard_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        refs: list[str] = []
        consumed: list[str] = []
        pending_refs: set[str] = set()
        outputs: list[CanonicalOutput] = []
        audit_entries: list[dict[str, Any]] = []
        pending_groups: list[PendingAggregation] = []

        for decision in plan.matches:
            output = self.patched_match_output(decision)
            decision_refs = [decision.left_id, decision.right_id]
            refs.extend(decision_refs)
            consumed.extend(decision_refs)
            outputs.append(output)
            audit_entries.append(
                {
                    "operation": "match",
                    "source_refs": decision_refs,
                    "base_ref": decision.base_ref,
                    "patch": decision.patch,
                    "outputs": [output.item.model_dump(mode="json")],
                }
            )

        self.validate_refs(plan.independent_ids)
        independent_outputs = [
            CanonicalOutput(kind=self.kinds[ref], item=self.unresolved[ref])
            for ref in plan.independent_ids
        ]
        refs.extend(plan.independent_ids)
        consumed.extend(plan.independent_ids)
        outputs.extend(independent_outputs)
        if plan.independent_ids:
            audit_entries.append(
                {
                    "operation": "independent",
                    "source_refs": plan.independent_ids,
                    "outputs": [
                        output.item.model_dump(mode="json")
                        for output in independent_outputs
                    ],
                }
            )

        for group in plan.resolutions:
            self.validate_refs(group.left_ids, "left")
            self.validate_refs(group.right_ids, "right")
            group_refs = set(group.left_ids + group.right_ids)
            refs.extend(group.left_ids + group.right_ids)
            covered: set[str] = set()
            for output in group.outputs:
                output_refs = set(output.source_refs)
                if len(output_refs) != len(output.source_refs) or not output_refs <= group_refs:
                    raise ValueError(f"{group.group_id}: invalid output source_refs")
                both_sides = bool(output_refs & set(group.left_ids)) and bool(
                    output_refs & set(group.right_ids)
                )
                if (output.alignment == "match") != both_sides:
                    raise ValueError(
                        f"{group.group_id}: invalid alignment for output sources"
                    )
                if any(not aspect.strip() for aspect in output.preserved_aspects):
                    raise ValueError(
                        f"{group.group_id}: preserved_aspects must be nonblank"
                    )
                covered.update(output_refs)
            if group.status == "resolved":
                if covered != group_refs:
                    raise ValueError(
                        f"{group.group_id}: every source must map to an output"
                    )
                consumed.extend(group.left_ids + group.right_ids)
                outputs.extend(group.outputs)
            else:
                pending_refs.update(group_refs)
                pending_groups.append(
                    PendingAggregation(
                        group_id=group.group_id,
                        left_source_item_ids=self.left.source_item_ids,
                        right_source_item_ids=self.right.source_item_ids,
                        left_ids=group.left_ids,
                        right_ids=group.right_ids,
                        reason=group.reason,
                        missing_evidence=group.missing_evidence,
                    )
                )
            audit_entries.append(
                {
                    "operation": "resolve_partial",
                    "group_id": group.group_id,
                    "status": group.status,
                    "source_refs": group.left_ids + group.right_ids,
                    "outputs": [
                        output.model_dump(mode="json") for output in group.outputs
                    ],
                    "reason": group.reason,
                    "missing_evidence": group.missing_evidence,
                }
            )

        self.validate_refs(refs)
        expected_refs = set(self.unresolved)
        if set(refs) != expected_refs:
            missing = sorted(expected_refs - set(refs))
            raise ValueError(f"aggregation plan must cover every unresolved source: {missing}")
        self.validate_output_ids(outputs)
        self.commit(consumed, outputs, audit_entries)

        remaining_group_ids = [
            group["group_id"]
            for group in hard_groups
            if set([*group["left_ids"], *group["right_ids"]]) & pending_refs
        ]
        processed_group_ids = [
            group["group_id"]
            for group in hard_groups
            if group["group_id"] not in remaining_group_ids
        ]
        return {
            "committed": True,
            "status": "pending" if pending_groups else "completed",
            "processed_group_ids": processed_group_ids,
            "remaining_group_ids": remaining_group_ids,
            "pending_groups": [
                group.model_dump(mode="json") for group in pending_groups
            ],
        }


def build_apply_aggregation_plan_tool(
    runtime: AggregationRuntime,
    hard_groups: list[dict[str, Any]],
) -> BaseTool:
    @tool("apply_aggregation_plan", args_schema=AggregationPlan)
    def apply_aggregation_plan(
        matches: list[MatchPatch],
        independent_ids: list[str],
        resolutions: list[PartialResolution],
    ) -> dict[str, Any]:
        """Atomically apply all remaining aggregation decisions in one plan."""
        plan = AggregationPlan(
            matches=matches,
            independent_ids=independent_ids,
            resolutions=resolutions,
        )
        try:
            return runtime.apply_plan(plan, hard_groups)
        except ValueError as exc:
            return {
                "committed": False,
                "errors": [str(exc)],
                "remaining_group_ids": [group["group_id"] for group in hard_groups],
            }

    return apply_aggregation_plan


class AggregationGraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    attempts: int
    outcome: NotRequired[MarketAggregationOutcome]


def build_market_aggregation_graph(
    *,
    model: BaseChatModel,
    runtime: AggregationRuntime,
    hard_groups: list[dict[str, Any]],
    max_attempts: int = CONFIG.market.max_aggregation_attempts,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    plan_tool = build_apply_aggregation_plan_tool(runtime, hard_groups)
    chat_model = model.bind_tools([plan_tool])
    system_message = SystemMessage(PROMPT_PATH.read_text(encoding="utf-8"))

    def call_model(state: AggregationGraphState) -> dict[str, Any]:
        response = chat_model.invoke([system_message, *state["messages"]])
        return {"messages": [response], "attempts": state["attempts"] + 1}

    def call_tools(state: AggregationGraphState) -> dict[str, Any]:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            raise RuntimeError("aggregation tools node requires an AIMessage")
        if any(call["name"] != "apply_aggregation_plan" for call in message.tool_calls):
            content = "No plan applied: use apply_aggregation_plan."
            return {
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                    for call in message.tool_calls
                ]
            }
        try:
            plans = [AggregationPlan.model_validate(call["args"]) for call in message.tool_calls]
            combined = AggregationPlan(
                matches=[decision for plan in plans for decision in plan.matches],
                independent_ids=[ref for plan in plans for ref in plan.independent_ids],
                resolutions=[group for plan in plans for group in plan.resolutions],
            )
            output = plan_tool.invoke(combined.model_dump(mode="python"))
        except ValueError as exc:
            output = {
                "committed": False,
                "errors": [str(exc)],
                "remaining_group_ids": [group["group_id"] for group in hard_groups],
            }
        content = json.dumps(output, ensure_ascii=False)
        update: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=call["id"],
                    name=call["name"],
                )
                for call in message.tool_calls
            ]
        }
        if output.get("committed"):
            if output["status"] == "pending":
                update["outcome"] = MarketAggregationOutcome(
                    status="pending",
                    pending_groups=output["pending_groups"],
                    unresolved=list(runtime.unresolved),
                    audit=runtime.audit,
                )
            else:
                update["outcome"] = MarketAggregationOutcome(
                    status="completed",
                    collection=runtime.collection(),
                    audit=runtime.audit,
                )
        return update

    def reject_plain_response(_state: AggregationGraphState) -> dict[str, Any]:
        return {
            "messages": [
                HumanMessage(
                    "Runtime constraint: submit one complete apply_aggregation_plan."
                )
            ]
        }

    def stop_at_limit(_state: AggregationGraphState) -> dict[str, Any]:
        return {
            "outcome": MarketAggregationOutcome(
                status="pending",
                pending_groups=[
                    PendingAggregation(
                        group_id="aggregation_retry_limit",
                        left_source_item_ids=runtime.left.source_item_ids,
                        right_source_item_ids=runtime.right.source_item_ids,
                        left_ids=[
                            ref for ref in runtime.unresolved if ref.startswith("left:")
                        ],
                        right_ids=[
                            ref for ref in runtime.unresolved if ref.startswith("right:")
                        ],
                        reason="Market aggregation did not produce a valid plan.",
                        missing_evidence=[
                            f"Aggregation retry limit reached after {max_attempts} attempts."
                        ],
                    )
                ],
                unresolved=list(runtime.unresolved),
                audit=runtime.audit,
            )
        }

    def after_model(state: AggregationGraphState) -> Literal["tools", "retry", "limit"]:
        message = state["messages"][-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            return "tools"
        return "limit" if state["attempts"] >= max_attempts else "retry"

    def after_tools(state: AggregationGraphState) -> Literal["end", "model", "limit"]:
        if state.get("outcome") is not None:
            return "end"
        return "limit" if state["attempts"] >= max_attempts else "model"

    def after_retry(state: AggregationGraphState) -> Literal["model", "limit"]:
        return "limit" if state["attempts"] >= max_attempts else "model"

    builder = StateGraph(AggregationGraphState)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_node("retry", reject_plain_response)
    builder.add_node("limit", stop_at_limit)
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        after_model,
        {"tools": "tools", "retry": "retry", "limit": "limit"},
    )
    builder.add_conditional_edges(
        "tools",
        after_tools,
        {"end": END, "model": "model", "limit": "limit"},
    )
    builder.add_conditional_edges(
        "retry",
        after_retry,
        {"model": "model", "limit": "limit"},
    )
    builder.add_edge("limit", END)
    return builder.compile(name="market_aggregation")


class MarketAggregationAgent:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def prepare(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
        report: EvalReport,
    ) -> tuple[AggregationRuntime, list[dict[str, Any]]]:
        validate_eval_report(report, left, right)
        runtime = AggregationRuntime(left, right)
        return runtime, runtime.apply_direct_groups(report)

    def graph_input(
        self,
        runtime: AggregationRuntime,
        hard_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        relevant_refs = {
            ref
            for group in hard_groups
            for ref in [*group["left_ids"], *group["right_ids"]]
        }
        context = {
            "groups": hard_groups,
            "items": {
                reference: item.model_dump(mode="json")
                for reference, item in runtime.unresolved.items()
                if reference in relevant_refs
            },
        }
        return {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False),
                }
            ],
            "attempts": 0,
        }

    def completed_outcome(self, runtime: AggregationRuntime) -> MarketAggregationOutcome:
        return MarketAggregationOutcome(
            status="completed",
            collection=runtime.collection(),
            audit=runtime.audit,
        )

    def invoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
        report: EvalReport,
    ) -> MarketAggregationOutcome:
        runtime, hard_groups = self.prepare(left, right, report)
        if not hard_groups:
            return self.completed_outcome(runtime)
        graph = build_market_aggregation_graph(
            model=self.model,
            runtime=runtime,
            hard_groups=hard_groups,
        )
        state = graph.invoke(self.graph_input(runtime, hard_groups))
        return MarketAggregationOutcome.model_validate(state["outcome"])

    async def ainvoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
        report: EvalReport,
    ) -> MarketAggregationOutcome:
        runtime, hard_groups = self.prepare(left, right, report)
        if not hard_groups:
            return self.completed_outcome(runtime)
        graph = build_market_aggregation_graph(
            model=self.model,
            runtime=runtime,
            hard_groups=hard_groups,
        )
        state = await graph.ainvoke(self.graph_input(runtime, hard_groups))
        return MarketAggregationOutcome.model_validate(state["outcome"])
