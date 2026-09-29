import asyncio

import pytest

from shop_agent_langgraph.agents.personalization.schemas import (
    PersonalizationContent, Constraint, validate_output, OutputValidationError,
)
from shop_agent_langgraph.agents.personalization.filtering import evaluate_constraint
from shop_agent_langgraph.agents.research.schemas import ResearchResult, Evidence
from test_relation_agent import profile, valid_graph
from test_market_aggregation import FixedMarketAgent, RecordingAggregator


def test_budget_without_shared_graph_dimension_is_executable():
    p = profile()
    before = p.model_dump()
    content = PersonalizationContent(context=[], criteria=[], attributes=[], unresolved=[],
        local_attributes=[dict(id="price", name="价格", description="商品到手价",
            aliases=[], type="numeric", units=["CNY"])], constraints=[dict(
                source_ref="local:price", operator="lte", values=[500], unit="CNY",
                strength="hard", user_quote="预算500内")], ready_to_search=True)
    validate_output(content, p, valid_graph(), ["预算500内"])
    assert evaluate_constraint(content.constraints[0], 499, "CNY") == "match"
    assert evaluate_constraint(content.constraints[0], 501, "CNY") == "no_match"
    assert evaluate_constraint(content.constraints[0], None, "CNY") == "unknown"
    assert evaluate_constraint(content.constraints[0], 50, "USD") == "unknown"
    assert p.model_dump() == before
    content.constraints[0].unit = "kg"
    with pytest.raises(OutputValidationError, match="matching units"):
        validate_output(content, p, valid_graph(), ["预算500内"])


def test_market_excludes_wrong_products_and_accessory_evidence():
    class ScopedMarket(FixedMarketAgent):
        async def _research(self, ids, target=""):
            return [ResearchResult(item_id="product_1", relevance="relevant", evidence=[
                Evidence(name="battery", source_text="10 h"),
                Evidence(name="clip", source_text="included clip", subject="accessory")]),
                ResearchResult(item_id="product_2", relevance="irrelevant", evidence=[
                    Evidence(name="fingers_supported", source_text="six fingers")])]
    aggregator = RecordingAggregator()
    result = asyncio.run(ScopedMarket(object(), aggregator).ainvoke("controller"))
    assert result.item_ids == ["product_1"]
    assert [e.name for e in aggregator.calls[0][0].evidence] == ["battery"]
    assert result.audit[1]["relevance"] == "irrelevant"


def test_uncertain_products_do_not_publish_market_profile():
    class UncertainMarket(FixedMarketAgent):
        async def _research(self, ids, target=""):
            return [ResearchResult(item_id="product_1", evidence=[])]
    aggregator = RecordingAggregator()
    result = asyncio.run(UncertainMarket(object(), aggregator).ainvoke("controller"))
    assert result.status == "pending"
    assert not aggregator.calls


def test_market_reviews_additions_without_changing_existing_definitions():
    from shop_agent_langgraph.agents.market.review import review_suggestions
    from shop_agent_langgraph.domain.market_cache import MarketCache
    from shop_agent_langgraph.agents.relation.schemas import ProfileSuggestion
    from test_relation_agent import ScriptedChatModel, call
    p = profile()
    g = valid_graph()
    g.profile_suggestions = [ProfileSuggestion(id="latency", kind="criterion",
        name="Latency", reason="Needed for timing", evidence_ids=["ev_one"])]
    addition = dict(kind="criterion", item=dict(id="input_latency", name="Latency",
        description="Input-to-response measured latency, mode-specific", aliases=[],
        type="numeric", units=["ms"], direction=dict(type="smaller_better")))
    model = ScriptedChatModel(responses=[call("Review", {"decisions": [dict(
        suggestion_id="latency", reason="Distinct measurement", addition=addition)]})])
    updated = review_suggestions(p, g, MarketCache.node("301"), model)
    assert len(p.criteria) == 1
    assert updated.criteria[0] == p.criteria[0]
    assert updated.criteria[-1].id == "input_latency"
    assert updated.audit[-1]["accepted"]


def test_personalization_narrows_candidate_scope_without_switching_knowledge(tmp_path):
    from test_personalization_agent import setup, questions, content
    from test_relation_agent import call
    payload = content() | {"candidate_scope": {"node_id": "543590", "user_quote": "无线"},
                           "ready_to_search": True}
    agent, _, _ = setup(tmp_path, [call("QuestionSet", questions()),
                                   call("PersonalizationContent", payload)])
    agent.prepare_questions("301", ["买手柄"], thread_id="a", task_id="t")
    result = agent.personalize("301", ["买手柄", "每天四小时，续航优先，无线"],
                               thread_id="a", task_id="t")
    assert result.node_id == "301"
    assert result.candidate_scope.node_id == "543590"
    assert result.ready_to_search and not result.blocking_questions


def test_relation_feedback_rebuilds_once_and_stops(tmp_path):
    from test_relation_agent import setup_agent, script, call, ScriptedChatModel
    suggestion = call("suggest_profile_change", {"suggestion": dict(id="latency",
        kind="criterion", name="Latency", reason="Different measurement",
        evidence_ids=["ev_one"])})
    first = script().responses
    first.insert(-2, suggestion)
    second = script().responses
    second.insert(-2, suggestion)
    review = call("Review", {"decisions": [dict(suggestion_id="latency", reason="Distinct",
        addition=dict(kind="criterion", item=dict(id="input_latency", name="Latency",
            description="Mode-specific input latency", aliases=[], type="numeric",
            units=["ms"], direction=dict(type="smaller_better"))))]})
    model = ScriptedChatModel(responses=first + [review] + second)
    agent, market, store, _ = setup_agent(tmp_path, model)
    result = agent.invoke("301")
    assert model.position == len(model.responses)
    assert "input_latency" in [c.id for c in market.load(market.node("301")).criteria]
    assert store.load("301") == result


def test_ready_state_rejects_blocking_questions():
    content = PersonalizationContent(context=[], criteria=[], attributes=[], unresolved=[],
                                    ready_to_search=True, blocking_questions=["Which platform?"])
    with pytest.raises(OutputValidationError, match="blocking"):
        validate_output(content, profile(), valid_graph(), ["controller"])
