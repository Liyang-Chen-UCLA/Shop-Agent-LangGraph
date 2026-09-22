import asyncio
import json

from shop_agent_langgraph.agents.market.aggregation import (
    MarketAggregationAgent,
    deterministic_premerge,
)
from shop_agent_langgraph.agents.market.graph import MarketAgent
from shop_agent_langgraph.agents.market.schemas import MarketSelection
from shop_agent_langgraph.domain.criteria import (
    BooleanAttribute,
    CategoricalAttribute,
    CriteriaAttributeSet,
)


def boolean_attribute(
    item_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    aliases: list[str] | None = None,
) -> BooleanAttribute:
    return BooleanAttribute(
        id=item_id,
        name=name or item_id,
        description=description or item_id,
        aliases=aliases or [],
        type="boolean",
    )


def collection(item_id: str, *attributes) -> CriteriaAttributeSet:
    return CriteriaAttributeSet(
        source_item_ids=[item_id],
        criteria=[],
        attributes=list(attributes),
    )


def test_premerge_removes_only_exactly_compatible_duplicates() -> None:
    inputs = [
        collection(
            "product_1",
            boolean_attribute(
                "wireless",
                name="Wireless",
                description="Supports wireless use.",
            ),
        ),
        collection(
            "product_2",
            boolean_attribute(
                "WIRELESS",
                name=" wireless ",
                description=" supports WIRELESS use. ",
                aliases=["cordless"],
            ),
        ),
    ]

    result = deterministic_premerge(inputs)

    assert len(result) == 2
    assert [len(value.attributes) for value in result] == [1, 0]
    assert result[0].attributes[0].aliases == ["cordless"]
    assert inputs[0].attributes[0].aliases == []


def test_premerge_keeps_similar_or_conflicting_items_independent() -> None:
    inputs = [
        collection("product_1", boolean_attribute("vibration", name="Vibration")),
        collection("product_2", boolean_attribute("rumble", name="Rumble")),
        collection(
            "product_3",
            CategoricalAttribute(
                id="vibration",
                name="Vibration",
                description="Vibration modes",
                aliases=[],
                type="categorical",
                values=["off", "low", "high"],
                value_domain="closed",
            ),
        ),
    ]

    result = deterministic_premerge(inputs)

    assert [len(value.attributes) for value in result] == [1, 1, 1]


def test_premerge_leaves_different_descriptions_for_the_model() -> None:
    inputs = [
        collection(
            "product_1",
            boolean_attribute("wireless", description="Supports Wi-Fi."),
        ),
        collection(
            "product_2",
            boolean_attribute("wireless", description="Supports Bluetooth."),
        ),
    ]

    result = deterministic_premerge(inputs)

    assert [len(value.attributes) for value in result] == [1, 1]


class RecordingAggregator:
    def __init__(self) -> None:
        self.calls: list[list[CriteriaAttributeSet]] = []

    async def ainvoke(
        self,
        collections: list[CriteriaAttributeSet],
    ) -> CriteriaAttributeSet:
        self.calls.append(collections)
        return CriteriaAttributeSet(
            source_item_ids=[
                item_id
                for collection in collections
                for item_id in collection.source_item_ids
            ],
            criteria=[item for collection in collections for item in collection.criteria],
            attributes=[
                item for collection in collections for item in collection.attributes
            ],
        )


class FixedMarketAgent(MarketAgent):
    async def _select(self, query: str) -> MarketSelection:
        return MarketSelection(item_ids=["product_1", "product_2"])

    async def _research(self, item_ids: list[str]) -> list[CriteriaAttributeSet]:
        return [
            collection("product_1", boolean_attribute("wireless", name="Wireless")),
            collection("product_2", boolean_attribute("wireless", name="Wireless")),
        ]


def test_market_premerges_then_aggregates_once_without_eval() -> None:
    aggregator = RecordingAggregator()
    agent = FixedMarketAgent(object(), aggregator)  # type: ignore[arg-type]

    result = asyncio.run(agent.ainvoke("controller"))

    assert result.status == "completed"
    assert len(aggregator.calls) == 1
    assert [len(value.attributes) for value in aggregator.calls[0]] == [1, 0]
    assert [value.id for value in result.attributes] == ["wireless"]


class StubAggregationGraph:
    def __init__(self, result: CriteriaAttributeSet) -> None:
        self.result = result
        self.inputs: list[dict[str, object]] = []

    def invoke(self, value):
        self.inputs.append(value)
        return {"structured_response": self.result}


def test_aggregation_input_contains_all_collections_and_preserves_source_ids() -> None:
    inputs = [
        collection("product_1", boolean_attribute("wireless")),
        collection("product_2", boolean_attribute("lighting")),
    ]
    graph = StubAggregationGraph(
        collection("hallucinated", boolean_attribute("connectivity"))
    )
    agent = MarketAggregationAgent.__new__(MarketAggregationAgent)
    agent.graph = graph  # type: ignore[assignment]

    result = agent.invoke(inputs)

    assert result.source_item_ids == ["product_1", "product_2"]
    message = graph.inputs[0]["messages"][0]  # type: ignore[index]
    payload = json.loads(message.content)
    assert len(payload["researched_collections"]) == 2


def test_aggregation_rejects_empty_input() -> None:
    try:
        MarketAggregationAgent._input([])
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty aggregation input should fail")
