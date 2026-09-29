import asyncio
import json

from shop_agent_langgraph.agents.market.aggregation import (
    MarketAggregationAgent,
    deterministic_premerge,
)
from shop_agent_langgraph.agents.market.graph import MarketAgent
from shop_agent_langgraph.agents.research.schemas import Evidence, ResearchResult
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


def research_result(item_id: str) -> ResearchResult:
    return ResearchResult(item_id=item_id, relevance="relevant", evidence=[Evidence(
        name="Polling rate", value="1000", unit="Hz", qualifier="Wired mode",
        source_text="Wired mode supports a polling rate of 1000 Hz.",
    )])


class RecordingAggregator:
    def __init__(self) -> None:
        self.calls: list[list[ResearchResult]] = []

    async def ainvoke(self, results: list[ResearchResult], target="") -> CriteriaAttributeSet:
        self.calls.append(results)
        return collection(results[0].item_id, boolean_attribute("wireless"))


class FixedMarketAgent(MarketAgent):
    async def _select(self, query: str, target="") -> MarketSelection:
        return MarketSelection(item_ids=["product_1", "product_2"])

    async def _research(self, item_ids: list[str], target="") -> list[ResearchResult]:
        return [
            research_result("product_1"),
            research_result("product_2"),
        ]


def test_market_passes_all_evidence_to_aggregation_once_without_eval() -> None:
    aggregator = RecordingAggregator()
    agent = FixedMarketAgent(object(), aggregator)  # type: ignore[arg-type]

    result = asyncio.run(agent.ainvoke("controller"))

    assert result.status == "completed"
    assert len(aggregator.calls) == 1
    assert [len(value.evidence) for value in aggregator.calls[0]] == [1, 1]
    assert [value.item_id for value in aggregator.calls[0]] == result.item_ids
    assert [value.id for value in result.attributes] == ["wireless"]


class StubAggregationGraph:
    def __init__(self, result: CriteriaAttributeSet) -> None:
        self.result = result
        self.inputs: list[dict[str, object]] = []

    def invoke(self, value):
        self.inputs.append(value)
        return {"structured_response": self.result}

    async def ainvoke(self, value):
        return self.invoke(value)


def test_aggregation_input_contains_all_evidence_and_preserves_source_ids() -> None:
    inputs = [
        research_result("product_1"),
        research_result("product_2"),
        ResearchResult(item_id="product_1", evidence=[]),
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
    assert payload["research_results"] == [value.model_dump() for value in inputs]


def test_aggregation_rejects_empty_input() -> None:
    try:
        MarketAggregationAgent._input([])
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty aggregation input should fail")


def test_empty_evidence_is_valid_and_keeps_product_provenance() -> None:
    inputs = [ResearchResult(item_id="product_1", evidence=[])]
    graph = StubAggregationGraph(collection("invented"))
    agent = MarketAggregationAgent.__new__(MarketAggregationAgent)
    agent.graph = graph
    result = agent.invoke(inputs)
    assert result.source_item_ids == ["product_1"]
    assert result.criteria == result.attributes == []


def test_evidence_contract_and_submission() -> None:
    from pydantic import ValidationError
    import pytest
    from shop_agent_langgraph.agents.research.tools import submit_research_result

    evidence = Evidence(name="Weight", source_text="Weight not specified")
    assert evidence.value is evidence.unit is evidence.qualifier is None
    result = submit_research_result.invoke({
        "item_id": "product_1", "evidence": [evidence.model_dump()],
    })
    assert ResearchResult.model_validate(result).evidence == [evidence]
    with pytest.raises(ValidationError):
        Evidence(name="Weight")
    with pytest.raises(ValidationError):
        ResearchResult(item_id="product_1", criteria=[], attributes=[])


def test_market_research_returns_evidence_unchanged(monkeypatch) -> None:
    import importlib
    module = importlib.import_module("shop_agent_langgraph.agents.market.graph")
    results = {item_id: research_result(item_id) for item_id in ["p1", "p2"]}

    async def research(item_id, target=""):
        return results[item_id]

    monkeypatch.setattr(module.research_agent, "ainvoke", research)
    agent = MarketAgent(object(), RecordingAggregator())
    assert asyncio.run(agent._research(["p1", "p2"])) == list(results.values())


def test_async_aggregation_preserves_evidence_and_parses_structured_response() -> None:
    inputs = [research_result("product_1")]
    graph = StubAggregationGraph(collection("invented"))
    graph.result = collection("invented").model_dump()
    agent = MarketAggregationAgent.__new__(MarketAggregationAgent)
    agent.graph = graph
    result = asyncio.run(agent.ainvoke(inputs))
    assert result.source_item_ids == ["product_1"]
    payload = json.loads(graph.inputs[0]["messages"][0].content)
    assert payload["research_results"] == [inputs[0].model_dump()]
