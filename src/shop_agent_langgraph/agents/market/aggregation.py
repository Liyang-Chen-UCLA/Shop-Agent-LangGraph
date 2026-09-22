from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, Sequence, TypedDict

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
    CanonicalOutput,
    FinalizeAggregation,
    IndependentBatch,
    MarketAggregationOutcome,
    MatchBatch,
    MatchDecision,
    PartialResolution,
    PartialResolutionBatch,
    PendingAggregation,
)


PROMPT_PATH = Path(__file__).with_name("aggregation_prompt.md")


def build_market_aggregation_tools(
    left: CriteriaAttributeSet,
    right: CriteriaAttributeSet,
) -> tuple[
    list[BaseTool],
    dict[str, Criterion | Attribute],
    list[Criterion],
    list[Attribute],
]:
    unresolved: dict[str, Criterion | Attribute] = {}
    kinds: dict[str, str] = {}
    for side, value in (("left", left), ("right", right)):
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
        """Atomically merge semantically equivalent left/right fields."""
        decisions = [MatchDecision.model_validate(value) for value in matches]
        refs: list[str] = []
        for decision in decisions:
            validate_refs([decision.left_id], "left")
            validate_refs([decision.right_id], "right")
            refs.extend([decision.left_id, decision.right_id])
        commit(refs, decisions)
        return {
            "committed": True,
            "accepted": len(decisions),
            "unresolved": list(unresolved),
        }

    @tool("independent", args_schema=IndependentBatch)
    def independent(item_ids: list[str]) -> dict[str, Any]:
        """Atomically retain distinct unresolved fields without rewriting them."""
        validate_refs(item_ids)
        outputs = [
            CanonicalOutput(kind=kinds[ref], item=unresolved[ref]) for ref in item_ids
        ]
        commit(item_ids, outputs)
        return {
            "committed": True,
            "accepted": len(item_ids),
            "unresolved": list(unresolved),
        }

    @tool("resolve_partial", args_schema=PartialResolutionBatch)
    def resolve_partial(resolutions: list[PartialResolution]) -> dict[str, Any]:
        """Resolve many-to-many overlaps or return explicit evidence requests."""
        try:
            groups = [PartialResolution.model_validate(value) for value in resolutions]
            if len({group.group_id for group in groups}) != len(groups):
                raise ValueError("group_id must be unique within the batch")
            all_refs: list[str] = []
            consumed: list[str] = []
            outputs: list[CanonicalOutput] = []
            results: list[dict[str, Any]] = []
            pending_groups: list[dict[str, Any]] = []
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
                        raise ValueError(
                            f"{group.group_id}: preserved_aspects must be nonblank"
                        )
                    covered.update(refs)
                if group.status == "resolved":
                    if covered != group_refs:
                        raise ValueError(
                            f"{group.group_id}: every source must map to an output"
                        )
                    consumed.extend(group.left_ids + group.right_ids)
                    outputs.extend(group.outputs)
                else:
                    pending_groups.append(
                        PendingAggregation(
                            group_id=group.group_id,
                            left_source_item_ids=left.source_item_ids,
                            right_source_item_ids=right.source_item_ids,
                            left_ids=group.left_ids,
                            right_ids=group.right_ids,
                            reason=group.reason,
                            missing_evidence=group.missing_evidence,
                        ).model_dump(mode="json")
                    )
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
                            output.model_dump(mode="json") for output in group.outputs
                        ],
                        "missing_evidence": group.missing_evidence,
                    }
                )
            validate_refs(all_refs)
            commit(consumed, outputs)
        except ValueError as exc:
            return {
                "committed": False,
                "status": "invalid",
                "errors": [str(exc)],
                "unresolved": list(unresolved),
            }
        return {
            "committed": True,
            "status": "pending" if pending_groups else "processed",
            "results": results,
            "pending_groups": pending_groups,
            "unresolved": list(unresolved),
        }

    @tool("submit_aggregation", args_schema=FinalizeAggregation)
    def submit_aggregation() -> dict[str, Any]:
        """Finalize from accepted operations only. Call alone with no arguments."""
        if unresolved:
            raise ValueError(f"unresolved item IDs remain: {list(unresolved)}")
        return CriteriaAttributeSet(
            source_item_ids=source_item_ids,
            criteria=criteria,
            attributes=attributes,
        ).model_dump(mode="json")

    return (
        [match, resolve_partial, independent, submit_aggregation],
        unresolved,
        criteria,
        attributes,
    )


class AggregationGraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    attempts: int
    outcome: NotRequired[MarketAggregationOutcome]


def build_market_aggregation_graph(
    *,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    unresolved: dict[str, Criterion | Attribute],
    left_source_item_ids: list[str],
    right_source_item_ids: list[str],
    max_attempts: int = CONFIG.market.max_aggregation_attempts,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    chat_model = model.bind_tools(tools)
    system_message = SystemMessage(PROMPT_PATH.read_text(encoding="utf-8"))
    tools_by_name = {aggregation_tool.name: aggregation_tool for aggregation_tool in tools}

    def call_model(state: AggregationGraphState) -> dict[str, Any]:
        response = chat_model.invoke([system_message, *state["messages"]])
        return {"messages": [response], "attempts": state["attempts"] + 1}

    def call_tools(state: AggregationGraphState) -> dict[str, Any]:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            raise RuntimeError("aggregation tools node requires an AIMessage")
        if len(message.tool_calls) != 1:
            return {
                "messages": [
                    ToolMessage(
                        content="No tools executed: call exactly one aggregation tool.",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                    for call in message.tool_calls
                ]
            }

        call = message.tool_calls[0]
        name = call["name"]
        arguments = call["args"]
        selected_tool = tools_by_name.get(name)
        if selected_tool is None:
            return {
                "messages": [
                    ToolMessage(
                        content=f"Unknown tool {name!r}. Use only the provided tools.",
                        tool_call_id=call["id"],
                        name=name,
                    )
                ]
            }

        try:
            if name == "submit_aggregation":
                selected_tool.get_input_schema().model_validate(arguments)
            output = selected_tool.invoke(arguments)
            content = (
                output
                if isinstance(output, str)
                else json.dumps(output, ensure_ascii=False)
            )
        except Exception as exc:
            return {
                "messages": [
                    ToolMessage(
                        content=f"Tool {name!r} failed: {exc}. Correct and retry.",
                        tool_call_id=call["id"],
                        name=name,
                    )
                ]
            }

        update: dict[str, Any] = {
            "messages": [ToolMessage(content=content, tool_call_id=call["id"], name=name)]
        }
        if name == "submit_aggregation":
            update["outcome"] = MarketAggregationOutcome(
                status="completed",
                collection=CriteriaAttributeSet.model_validate(output),
            )
        elif isinstance(output, dict) and output.get("status") == "pending":
            update["outcome"] = MarketAggregationOutcome(
                status="pending",
                pending_groups=output["pending_groups"],
                unresolved=output["unresolved"],
            )
        return update

    def reject_plain_response(_state: AggregationGraphState) -> dict[str, Any]:
        return {
            "messages": [
                HumanMessage(
                    "Runtime constraint: call one aggregation tool; plain text cannot finish."
                )
            ]
        }

    def stop_at_limit(_state: AggregationGraphState) -> dict[str, Any]:
        left_ids = [ref for ref in unresolved if ref.startswith("left:")]
        right_ids = [ref for ref in unresolved if ref.startswith("right:")]
        return {
            "outcome": MarketAggregationOutcome(
                status="pending",
                pending_groups=[
                    PendingAggregation(
                        group_id="aggregation_retry_limit",
                        left_source_item_ids=left_source_item_ids,
                        right_source_item_ids=right_source_item_ids,
                        left_ids=left_ids,
                        right_ids=right_ids,
                        reason="Market aggregation did not reach a validated result.",
                        missing_evidence=[
                            f"Aggregation retry limit reached after {max_attempts} attempts."
                        ],
                    )
                ],
                unresolved=list(unresolved),
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

    def _build_run(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
        report: EvalReport,
    ) -> tuple[CompiledStateGraph[Any, Any, Any, Any], dict[str, Any]]:
        validate_eval_report(report, left, right)
        tools, unresolved, _criteria, _attributes = build_market_aggregation_tools(
            left, right
        )
        graph = build_market_aggregation_graph(
            model=self.model,
            tools=tools,
            unresolved=unresolved,
            left_source_item_ids=left.source_item_ids,
            right_source_item_ids=right.source_item_ids,
        )
        context = {
            "left_source_item_ids": left.source_item_ids,
            "right_source_item_ids": right.source_item_ids,
            "items": {
                reference: item.model_dump(mode="json")
                for reference, item in unresolved.items()
            },
            "eval_report": report.model_dump(mode="json"),
        }
        return graph, {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False),
                }
            ],
            "attempts": 0,
        }

    def invoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
        report: EvalReport,
    ) -> MarketAggregationOutcome:
        graph, graph_input = self._build_run(left, right, report)
        state = graph.invoke(graph_input)
        return MarketAggregationOutcome.model_validate(state["outcome"])

    async def ainvoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
        report: EvalReport,
    ) -> MarketAggregationOutcome:
        graph, graph_input = self._build_run(left, right, report)
        state = await graph.ainvoke(graph_input)
        return MarketAggregationOutcome.model_validate(state["outcome"])
