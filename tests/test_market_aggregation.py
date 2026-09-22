import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from shop_agent_langgraph.agents.eval.schemas import EvalGroup, EvalReport
from shop_agent_langgraph.agents.market.aggregation import (
    AggregationRuntime,
    MarketAggregationAgent,
    build_market_aggregation_graph,
)
from shop_agent_langgraph.agents.market.graph import MarketAgent
from shop_agent_langgraph.agents.market.schemas import (
    MarketAggregationOutcome,
    MarketSelection,
    PendingAggregation,
)
from shop_agent_langgraph.domain.criteria import (
    BooleanAttribute,
    CategoricalAttribute,
    CriteriaAttributeSet,
)


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
                attribute("vibration", "Has adjustable vibration"),
                attribute("lighting", "Has decorative lighting"),
            ],
        ),
        CriteriaAttributeSet(
            source_item_ids=["product_2", "product_1"],
            criteria=[],
            attributes=[
                attribute("rumble", "Has vibration feedback"),
                attribute("light_bar", "Has a dynamic decorative light bar"),
            ],
        ),
    )


L1, L2 = "left:attribute:vibration", "left:attribute:lighting"
R1, R2 = "right:attribute:rumble", "right:attribute:light_bar"


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


def eval_report(status: str = "uncertain") -> EvalReport:
    return EvalReport(
        groups=[
            EvalGroup(
                status=status,
                left_ids=[L1, L2],
                right_ids=[R1, R2],
                reason="The definitions overlap at different granularity.",
            )
        ]
    )


def direct_report() -> EvalReport:
    return EvalReport(
        groups=[
            EvalGroup(
                status="match",
                left_ids=[L1],
                right_ids=[R1],
                reason="Same feature.",
            ),
            EvalGroup(
                status="independent",
                left_ids=[L2],
                right_ids=[],
                reason="Only on the left.",
            ),
            EvalGroup(
                status="independent",
                left_ids=[],
                right_ids=[R2],
                reason="Only on the right.",
            ),
        ]
    )


def resolved_group() -> dict[str, object]:
    return {
        "group_id": "group_0",
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
            output("dynamic_light_bar", [R2], "independent"),
        ],
    }


def call(args: dict[str, object], call_id: str = "plan") -> dict[str, object]:
    return {
        "name": "apply_aggregation_plan",
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


class ScriptedModel:
    def __init__(self, calls: list[list[dict[str, object]]]) -> None:
        self.calls = iter(calls)
        self.messages_seen: list[object] = []

    def bind_tools(self, tools):
        self.tools = {value.name for value in tools}
        return self

    def invoke(self, messages):
        self.messages_seen.append(messages)
        return AIMessage(content="", tool_calls=next(self.calls))


class FailIfCalledModel:
    def bind_tools(self, tools):
        raise AssertionError("ordinary groups must not invoke the aggregation model")


def test_ordinary_match_and_independent_skip_model() -> None:
    outcome = MarketAggregationAgent(FailIfCalledModel()).invoke(  # type: ignore[arg-type]
        *collections(), direct_report()
    )

    assert outcome.status == "completed"
    assert outcome.collection is not None
    assert outcome.collection.source_item_ids == ["product_1", "product_2"]
    assert {value.id for value in outcome.collection.attributes} == {
        "vibration",
        "lighting",
        "light_bar",
    }
    merged = next(value for value in outcome.collection.attributes if value.id == "vibration")
    assert merged.aliases == ["rumble"]
    assert [entry["operation"] for entry in outcome.audit] == [
        "match",
        "independent",
        "independent",
    ]


def test_one_mixed_plan_auto_finalizes_without_submit_round() -> None:
    model = ScriptedModel(
        [[call({
            "matches": [{
                "left_id": L1,
                "right_id": R1,
                "base_ref": L1,
                "patch": {
                    "id": "haptics",
                    "description": "Canonical haptic support",
                    "aliases": ["rumble"],
                },
            }],
            "independent_ids": [L2, R2],
            "resolutions": [],
        })]]
    )

    outcome = MarketAggregationAgent(model).invoke(  # type: ignore[arg-type]
        *collections(), eval_report()
    )

    assert outcome.status == "completed"
    assert outcome.collection is not None
    assert {value.id for value in outcome.collection.attributes} == {
        "haptics", "lighting", "light_bar"
    }
    haptics = next(value for value in outcome.collection.attributes if value.id == "haptics")
    assert haptics.description == "Canonical haptic support"
    assert len(model.messages_seen) == 1
    assert model.tools == {"apply_aggregation_plan"}


def test_multiple_plan_calls_are_combined_atomically() -> None:
    model = ScriptedModel(
        [[
            call({
                "matches": [{
                    "left_id": L1,
                    "right_id": R1,
                    "base_ref": L1,
                    "patch": {"id": "haptics"},
                }],
                "independent_ids": [],
                "resolutions": [],
            }, "match"),
            call({
                "matches": [],
                "independent_ids": [L2, R2],
                "resolutions": [],
            }, "independent"),
        ]]
    )

    outcome = MarketAggregationAgent(model).invoke(  # type: ignore[arg-type]
        *collections(), eval_report()
    )

    assert outcome.status == "completed"
    assert outcome.collection is not None
    assert len(model.messages_seen) == 1


def test_incompatible_match_is_sent_to_model() -> None:
    left = CriteriaAttributeSet(
        source_item_ids=["left_product"],
        criteria=[],
        attributes=[attribute("feature")],
    )
    right = CriteriaAttributeSet(
        source_item_ids=["right_product"],
        criteria=[],
        attributes=[
            CategoricalAttribute(
                id="feature_mode",
                name="feature mode",
                description="Feature mode",
                aliases=[],
                type="categorical",
                values=["off", "on"],
                value_domain="closed",
            )
        ],
    )
    report = EvalReport(
        groups=[
            EvalGroup(
                status="match",
                left_ids=["left:attribute:feature"],
                right_ids=["right:attribute:feature_mode"],
                reason="Same concept with incompatible schemas.",
            )
        ]
    )
    model = ScriptedModel(
        [[call({
            "matches": [{
                "left_id": "left:attribute:feature",
                "right_id": "right:attribute:feature_mode",
                "base_ref": "right:attribute:feature_mode",
                "patch": {},
            }],
            "independent_ids": [],
            "resolutions": [],
        })]]
    )

    outcome = MarketAggregationAgent(model).invoke(left, right, report)  # type: ignore[arg-type]

    assert outcome.status == "completed"
    assert len(model.messages_seen) == 1
    assert isinstance(outcome.collection.attributes[0], CategoricalAttribute)  # type: ignore[union-attr]


def test_partial_plan_executes_and_tool_feedback_is_compact() -> None:
    model = ScriptedModel(
        [[call({"matches": [], "independent_ids": [], "resolutions": [resolved_group()]})]]
    )
    runtime = AggregationRuntime(*collections())
    hard_groups = runtime.apply_direct_groups(eval_report())
    graph = build_market_aggregation_graph(
        model=model,  # type: ignore[arg-type]
        runtime=runtime,
        hard_groups=hard_groups,
    )

    state = graph.invoke(MarketAggregationAgent(model).graph_input(runtime, hard_groups))  # type: ignore[arg-type]
    outcome = MarketAggregationOutcome.model_validate(state["outcome"])
    tool_message = next(message for message in state["messages"] if isinstance(message, ToolMessage))

    assert outcome.status == "completed"
    assert outcome.collection is not None and len(outcome.collection.attributes) == 4
    feedback = json.loads(tool_message.content)
    assert set(feedback) == {
        "committed", "status", "processed_group_ids", "remaining_group_ids", "pending_groups"
    }
    assert "description" not in tool_message.content
    assert outcome.audit[0]["outputs"]


def test_invalid_plan_is_atomic_then_can_retry() -> None:
    invalid = {
        "matches": [
            {"left_id": L1, "right_id": R1, "base_ref": L1, "patch": {"id": "duplicate"}},
            {"left_id": L2, "right_id": R2, "base_ref": L2, "patch": {"id": "duplicate"}},
        ],
        "independent_ids": [],
        "resolutions": [],
    }
    valid = {"matches": [], "independent_ids": [], "resolutions": [resolved_group()]}
    model = ScriptedModel([[call(invalid, "bad")], [call(valid, "good")]])

    outcome = MarketAggregationAgent(model).invoke(  # type: ignore[arg-type]
        *collections(), eval_report()
    )

    assert outcome.status == "completed"
    assert outcome.collection is not None and len(outcome.collection.attributes) == 4
    assert len(model.messages_seen) == 2
    error = next(
        message for message in model.messages_seen[1] if isinstance(message, ToolMessage)
    )
    assert "canonical output IDs must be unique" in error.content


def test_needs_review_stops_after_one_plan() -> None:
    pending = resolved_group()
    pending.update(
        status="needs_review",
        outputs=[],
        missing_evidence=["Need component-level evidence."],
    )
    model = ScriptedModel(
        [[call({"matches": [], "independent_ids": [], "resolutions": [pending]})]]
    )

    outcome = MarketAggregationAgent(model).invoke(  # type: ignore[arg-type]
        *collections(), eval_report()
    )

    assert outcome.status == "pending"
    assert outcome.pending_groups[0].missing_evidence == ["Need component-level evidence."]
    assert len(model.messages_seen) == 1


class PlainResponseModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content="still reasoning", tool_calls=[])


def test_retry_limit_returns_pending() -> None:
    runtime = AggregationRuntime(*collections())
    hard_groups = runtime.apply_direct_groups(eval_report())
    model = PlainResponseModel()
    graph = build_market_aggregation_graph(
        model=model,  # type: ignore[arg-type]
        runtime=runtime,
        hard_groups=hard_groups,
        max_attempts=2,
    )

    state = graph.invoke(MarketAggregationAgent(model).graph_input(runtime, hard_groups))  # type: ignore[arg-type]
    outcome = MarketAggregationOutcome.model_validate(state["outcome"])

    assert outcome.status == "pending"
    assert outcome.pending_groups[0].group_id == "aggregation_retry_limit"
    assert model.calls == 2


def test_duplicate_inputs_are_rejected_before_state_creation() -> None:
    left, right = collections()
    left.attributes.append(left.attributes[0])
    with pytest.raises(ValueError, match="duplicate input"):
        AggregationRuntime(left, right)


def test_pair_runtime_state_is_isolated() -> None:
    first = AggregationRuntime(*collections())
    second = AggregationRuntime(*collections())
    first.apply_direct_groups(direct_report())

    assert not first.unresolved
    assert set(second.unresolved) == {L1, L2, R1, R2}


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
                        left_ids=[f"left:attribute:{value.id}"],
                        right_ids=[],
                        reason="No corresponding right field.",
                    )
                    for value in left.attributes
                ],
                *[
                    EvalGroup(
                        status="independent",
                        left_ids=[],
                        right_ids=[f"right:attribute:{value.id}"],
                        reason="No corresponding left field.",
                    )
                    for value in right.attributes
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
                audit=[{"pair": [*left.source_item_ids, *right.source_item_ids]}],
            )
        return MarketAggregationOutcome(
            status="completed",
            collection=CriteriaAttributeSet(
                source_item_ids=list(dict.fromkeys([*left.source_item_ids, *right.source_item_ids])),
                criteria=[*left.criteria, *right.criteria],
                attributes=[*left.attributes, *right.attributes],
            ),
            audit=[{"pair": [*left.source_item_ids, *right.source_item_ids]}],
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
    assert len(outcome.audit) == 3


def test_pending_pair_stops_later_reduction_rounds() -> None:
    evaluator = RecordingEvaluator()
    aggregator = RecordingAggregator(pending_source="product_0")
    agent = MarketAgent(object(), evaluator, aggregator)  # type: ignore[arg-type]

    outcome = asyncio.run(agent._aggregate([singleton(index) for index in range(4)]))

    assert outcome.status == "pending"
    assert len(evaluator.calls) == 2
    assert len(aggregator.calls) == 2
    assert len(outcome.audit) == 2


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
    assert result.audit
