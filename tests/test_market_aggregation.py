import asyncio

import pytest
from langchain_core.messages import AIMessage

from shop_agent_langgraph.agents.eval.schemas import EvalGroup, EvalReport
from shop_agent_langgraph.agents.market.aggregation import (
    MarketAggregationAgent,
    build_market_aggregation_graph,
    build_market_aggregation_tools,
)
from shop_agent_langgraph.agents.market.graph import MarketAgent
from shop_agent_langgraph.agents.market.schemas import (
    MarketAggregationOutcome,
    MarketSelection,
    PendingAggregation,
)
from shop_agent_langgraph.domain.criteria import BooleanAttribute, CriteriaAttributeSet


def attribute(item_id: str, description: str | None = None) -> BooleanAttribute:
    return BooleanAttribute(
        id=item_id,
        name=item_id,
        description=description or item_id,
        aliases=[],
        type="boolean",
    )


def collections() -> tuple[CriteriaAttributeSet, CriteriaAttributeSet]:
    return (
        CriteriaAttributeSet(
            source_item_ids=["product_1"],
            criteria=[],
            attributes=[
                attribute("vibration_bundle", "Has adjustable vibration"),
                attribute("lighting", "Has decorative lighting"),
            ],
        ),
        CriteriaAttributeSet(
            source_item_ids=["product_2", "product_1"],
            criteria=[],
            attributes=[
                attribute("rumble", "Has vibration feedback"),
                attribute("dynamic_light_bar", "Has a dynamic decorative light bar"),
            ],
        ),
    )


L1, L2 = "left:attribute:vibration_bundle", "left:attribute:lighting"
R1, R2 = "right:attribute:rumble", "right:attribute:dynamic_light_bar"


def item(item_id: str) -> dict[str, object]:
    return {
        "id": item_id,
        "name": item_id,
        "description": item_id,
        "aliases": [],
        "type": "boolean",
    }


def output(
    item_id: str,
    refs: list[str],
    alignment: str = "match",
) -> dict[str, object]:
    return {
        "alignment": alignment,
        "kind": "attribute",
        "item": item(item_id),
        "source_refs": refs,
        "preserved_aspects": [item_id],
    }


def resolved_group() -> dict[str, object]:
    return {
        "group_id": "features",
        "left_ids": [L1, L2],
        "right_ids": [R1, R2],
        "relation": "overlap",
        "reason": "Overlapping existence and specialized capabilities",
        "status": "resolved",
        "missing_evidence": [],
        "outputs": [
            output("has_vibration", [L1, R1]),
            output("vibration_adjustable", [L1], "independent"),
            output("has_lighting", [L2, R2]),
            output("has_dynamic_light_bar", [R2], "independent"),
        ],
    }


def eval_report() -> EvalReport:
    return EvalReport(
        groups=[
            EvalGroup(
                status="uncertain",
                left_ids=[L1, L2],
                right_ids=[R1, R2],
                reason="The definitions overlap at different granularity.",
            )
        ]
    )


def call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class ScriptedModel:
    def __init__(self, calls: list[list[dict[str, object]]]) -> None:
        self.calls = iter(calls)
        self.messages_seen: list[object] = []

    def bind_tools(self, tools):
        self.tools = {tool.name for tool in tools}
        return self

    def invoke(self, messages):
        self.messages_seen.append(messages)
        return AIMessage(content="", tool_calls=next(self.calls))


def setup_tools():
    tools, unresolved, criteria, attributes = build_market_aggregation_tools(
        *collections()
    )
    return {tool.name: tool for tool in tools}, unresolved, criteria, attributes


def test_market_tools_execute_many_to_many_and_runtime_submission() -> None:
    tools, unresolved, criteria, attributes = setup_tools()
    result = tools["resolve_partial"].invoke(
        {"resolutions": [resolved_group()]}
    )
    assert result["committed"] is True
    assert result["status"] == "processed"
    assert not unresolved and not criteria and len(attributes) == 4
    final = tools["submit_aggregation"].invoke({})
    assert final["source_item_ids"] == ["product_1", "product_2"]
    assert {value["id"] for value in final["attributes"]} == {
        "has_vibration",
        "vibration_adjustable",
        "has_lighting",
        "has_dynamic_light_bar",
    }


def test_invalid_market_batch_is_atomic() -> None:
    tools, unresolved, criteria, attributes = setup_tools()
    before = dict(unresolved)
    with pytest.raises(ValueError):
        tools["match"].invoke(
            {
                "matches": [
                    {
                        "left_id": L1,
                        "right_id": R1,
                        "kind": "attribute",
                        "item": item("vibration"),
                    },
                    {
                        "left_id": L2,
                        "right_id": "right:attribute:missing",
                        "kind": "attribute",
                        "item": item("lighting"),
                    },
                ]
            }
        )
    assert unresolved == before and not criteria and not attributes


def test_conflicting_output_ids_do_not_partially_mutate_state() -> None:
    tools, unresolved, criteria, attributes = setup_tools()
    before = dict(unresolved)
    with pytest.raises(ValueError, match="canonical output IDs"):
        tools["match"].invoke(
            {
                "matches": [
                    {
                        "left_id": L1,
                        "right_id": R1,
                        "kind": "attribute",
                        "item": item("duplicate"),
                    },
                    {
                        "left_id": L2,
                        "right_id": R2,
                        "kind": "attribute",
                        "item": item("duplicate"),
                    },
                ]
            }
        )
    assert unresolved == before and not criteria and not attributes


def test_submit_rejects_unprocessed_sources() -> None:
    tools, unresolved, criteria, attributes = setup_tools()
    before = dict(unresolved)
    with pytest.raises(ValueError, match="unresolved"):
        tools["submit_aggregation"].invoke({})
    assert unresolved == before and not criteria and not attributes


def test_duplicate_inputs_are_rejected_before_state_creation() -> None:
    left, right = collections()
    left.attributes.append(left.attributes[0])
    with pytest.raises(ValueError, match="duplicate input"):
        build_market_aggregation_tools(left, right)


def test_pair_tool_state_is_isolated() -> None:
    first_tools, first_unresolved, _, _ = setup_tools()
    _second_tools, second_unresolved, _, _ = setup_tools()
    second_before = dict(second_unresolved)

    first_tools["independent"].invoke({"item_ids": [L1]})

    assert L1 not in first_unresolved
    assert second_unresolved == second_before


def test_market_aggregation_graph_builds_result_from_tool_state() -> None:
    model = ScriptedModel(
        [
            [
                call(
                    "resolve_partial",
                    {"resolutions": [resolved_group()]},
                    "resolve",
                )
            ],
            [call("submit_aggregation", {}, "submit")],
        ]
    )

    outcome = MarketAggregationAgent(model).invoke(  # type: ignore[arg-type]
        *collections(), eval_report()
    )

    assert outcome.status == "completed"
    assert outcome.collection is not None
    assert len(outcome.collection.attributes) == 4


def test_uncertain_report_does_not_force_partial_resolution() -> None:
    model = ScriptedModel(
        [
            [
                call(
                    "match",
                    {
                        "matches": [
                            {
                                "left_id": L1,
                                "right_id": R1,
                                "kind": "attribute",
                                "item": item("has_vibration"),
                            }
                        ]
                    },
                    "match",
                )
            ],
            [
                call(
                    "independent",
                    {"item_ids": [L2, R2]},
                    "independent",
                )
            ],
            [call("submit_aggregation", {}, "submit")],
        ]
    )

    outcome = MarketAggregationAgent(model).invoke(  # type: ignore[arg-type]
        *collections(), eval_report()
    )

    assert outcome.status == "completed"
    assert outcome.collection is not None
    assert {item.id for item in outcome.collection.attributes} == {
        "has_vibration",
        "lighting",
        "dynamic_light_bar",
    }


def test_market_aggregation_rejects_model_authored_final_result() -> None:
    model = ScriptedModel(
        [
            [
                call(
                    "resolve_partial",
                    {"resolutions": [resolved_group()]},
                    "resolve",
                )
            ],
            [
                call(
                    "submit_aggregation",
                    {
                        "source_item_ids": ["forged"],
                        "criteria": [],
                        "attributes": [],
                    },
                    "forged",
                )
            ],
            [call("submit_aggregation", {}, "submit")],
        ]
    )

    outcome = MarketAggregationAgent(model).invoke(  # type: ignore[arg-type]
        *collections(), eval_report()
    )

    assert outcome.status == "completed"
    assert outcome.collection is not None
    assert outcome.collection.source_item_ids == ["product_1", "product_2"]
    assert len(model.messages_seen) == 3


def test_needs_review_returns_pending_without_retrying() -> None:
    pending_group = resolved_group()
    pending_group.update(
        status="needs_review",
        outputs=[],
        missing_evidence=["Need component-level haptic evidence."],
    )
    model = ScriptedModel(
        [
            [
                call(
                    "resolve_partial",
                    {"resolutions": [pending_group]},
                    "pending",
                )
            ]
        ]
    )

    outcome = MarketAggregationAgent(model).invoke(  # type: ignore[arg-type]
        *collections(), eval_report()
    )

    assert outcome.status == "pending"
    assert outcome.pending_groups[0].missing_evidence == [
        "Need component-level haptic evidence."
    ]
    assert len(model.messages_seen) == 1


class PlainResponseModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content="still reasoning", tool_calls=[])


def test_aggregation_retry_limit_returns_pending() -> None:
    tools, unresolved, _criteria, _attributes = build_market_aggregation_tools(
        *collections()
    )
    model = PlainResponseModel()
    graph = build_market_aggregation_graph(
        model=model,  # type: ignore[arg-type]
        tools=tools,
        unresolved=unresolved,
        left_source_item_ids=["product_1"],
        right_source_item_ids=["product_2", "product_1"],
        max_attempts=2,
    )

    state = graph.invoke(
        {"messages": [{"role": "user", "content": "aggregate"}], "attempts": 0}
    )
    outcome = MarketAggregationOutcome.model_validate(state["outcome"])

    assert outcome.status == "pending"
    assert outcome.pending_groups[0].group_id == "aggregation_retry_limit"
    assert model.calls == 2


def singleton(index: int) -> CriteriaAttributeSet:
    return CriteriaAttributeSet(
        source_item_ids=[f"product_{index}"],
        criteria=[],
        attributes=[attribute(f"feature_{index}")],
    )


class RecordingEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    async def ainvoke(self, left, right):
        self.calls.append((left.source_item_ids, right.source_item_ids))
        return EvalReport(
            groups=[
                *[
                    EvalGroup(
                        status="independent",
                        left_ids=[f"left:attribute:{item.id}"],
                        right_ids=[],
                        reason="No corresponding right field.",
                    )
                    for item in left.attributes
                ],
                *[
                    EvalGroup(
                        status="independent",
                        left_ids=[],
                        right_ids=[f"right:attribute:{item.id}"],
                        reason="No corresponding left field.",
                    )
                    for item in right.attributes
                ],
            ]
        )


class RecordingAggregator:
    def __init__(self, pending_source: str | None = None) -> None:
        self.calls: list[tuple[list[str], list[str], EvalReport]] = []
        self.pending_source = pending_source

    async def ainvoke(self, left, right, report):
        self.calls.append((left.source_item_ids, right.source_item_ids, report))
        if self.pending_source in left.source_item_ids:
            return MarketAggregationOutcome(
                status="pending",
                pending_groups=[
                    PendingAggregation(
                        group_id="needs_evidence",
                        left_source_item_ids=left.source_item_ids,
                        right_source_item_ids=right.source_item_ids,
                        left_ids=[f"left:attribute:{left.attributes[0].id}"],
                        right_ids=[f"right:attribute:{right.attributes[0].id}"],
                        reason="Evidence is insufficient.",
                        missing_evidence=["Need comparable definitions."],
                    )
                ],
                unresolved=[
                    f"left:attribute:{left.attributes[0].id}",
                    f"right:attribute:{right.attributes[0].id}",
                ],
            )
        return MarketAggregationOutcome(
            status="completed",
            collection=CriteriaAttributeSet(
                source_item_ids=list(
                    dict.fromkeys([*left.source_item_ids, *right.source_item_ids])
                ),
                criteria=[*left.criteria, *right.criteria],
                attributes=[*left.attributes, *right.attributes],
            ),
        )


def test_market_reduction_evaluates_then_aggregates_each_pair() -> None:
    evaluator = RecordingEvaluator()
    aggregator = RecordingAggregator()
    agent = MarketAgent(object(), evaluator, aggregator)  # type: ignore[arg-type]

    outcome = asyncio.run(agent._aggregate([singleton(index) for index in range(4)]))

    assert outcome.status == "completed"
    assert outcome.collection is not None
    assert len(outcome.collection.source_item_ids) == 4
    assert len(evaluator.calls) == 3
    assert len(aggregator.calls) == 3
    assert all(isinstance(call[2], EvalReport) for call in aggregator.calls)


def test_pending_pair_stops_later_reduction_rounds() -> None:
    evaluator = RecordingEvaluator()
    aggregator = RecordingAggregator(pending_source="product_0")
    agent = MarketAgent(object(), evaluator, aggregator)  # type: ignore[arg-type]

    outcome = asyncio.run(agent._aggregate([singleton(index) for index in range(4)]))

    assert outcome.status == "pending"
    assert len(evaluator.calls) == 2
    assert len(aggregator.calls) == 2


class FixedMarketAgent(MarketAgent):
    async def _select(self, query: str) -> MarketSelection:
        return MarketSelection(item_ids=["product_0", "product_1"])

    async def _research(self, item_ids: list[str]) -> list[CriteriaAttributeSet]:
        return [singleton(0), singleton(1)]


def test_public_market_result_exposes_pending_status() -> None:
    agent = FixedMarketAgent(
        object(),  # type: ignore[arg-type]
        RecordingEvaluator(),  # type: ignore[arg-type]
        RecordingAggregator(pending_source="product_0"),  # type: ignore[arg-type]
    )

    result = asyncio.run(agent.ainvoke("controller"))

    assert result.status == "pending"
    assert result.criteria == [] and result.attributes == []
    assert result.pending_groups[0].group_id == "needs_evidence"
