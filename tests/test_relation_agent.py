import asyncio
import importlib
import json
from datetime import datetime, timezone

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, ValidationError

from shop_agent_langgraph.agents.market.schemas import MarketResult
from shop_agent_langgraph.agents.relation.draft import RelationDraft
from shop_agent_langgraph.agents.relation.graph import RelationAgent
from shop_agent_langgraph.agents.relation.schemas import (
    Condition, ContextVariable, DerivedVariable, OpenQuestion, Relation,
    RelationEvidence, RelationGraph, UtilityRule, profile_hash, validate_graph,
)
from shop_agent_langgraph.agents.relation.store import RelationStore, RevisionConflict
from shop_agent_langgraph.agents.relation.tools import build_relation_tools
from shop_agent_langgraph.domain.market_cache import MarketCache

NOW = datetime.now(timezone.utc)
URL = "https://example.org/study"
QUOTE = "Under the same workload, higher power consumption reduces battery runtime."


def profile():
    return MarketResult(query="controller", item_ids=["p1"], status="completed",
        criteria=[dict(id="battery_life", name="Battery life", description="Runtime", aliases=[],
                       type="numeric", units=["h"], direction={"type": "larger_better"})],
        attributes=[dict(id="power", name="Power", description="Operating power", aliases=[],
                         type="numeric", units=["W"])])


def empty_graph():
    return RelationGraph(node_id="301", profile_hash=profile_hash(profile()), revision=1,
                         created_at=NOW, updated_at=NOW)


def evidence(id="ev_one"):
    return RelationEvidence(id=id, url=URL, title="Study", source_text=QUOTE, retrieved_at=NOW,
                            source_kind="raw_content", scope="same workload")


def relation(**changes):
    fields = dict(id="power_runtime", source="attribute:power", target="criterion:battery_life",
                  kind="influence", effect="negative", statement=QUOTE,
                  evidence_ids=["ev_one"], status="supported")
    return Relation(**(fields | changes))


def valid_graph():
    graph = empty_graph()
    graph.evidence = [evidence()]
    graph.relations = [relation()]
    return graph


class Search:
    def __init__(self):
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return {"results": [{"url": URL, "title": "Study", "raw_content": QUOTE}]}


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    position: int = 0
    bound_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-relation-test"

    def bind_tools(self, tools, **kwargs):
        self.bound_names = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


def script():
    return ScriptedChatModel(responses=[
        call("search_relation_evidence", {"query": "power battery runtime controlled study"}),
        call("add_evidence", {"id": "ev_one", "url": URL, "source_text": QUOTE, "scope": "same workload"}),
        call("upsert_relation", {"relation": relation().model_dump(mode="json")}),
        call("finalize_relation_graph", {}),
        AIMessage(content="Done"),
    ])


def setup_agent(tmp_path, model=None):
    market = MarketCache(tmp_path / "market")
    market.save(market.node("301"), profile())
    store = RelationStore(tmp_path / "relations")
    search = Search()
    agent = RelationAgent(model or script(), store=store, market_cache=market, search=search)
    return agent, market, store, search


def test_real_tool_loop_searches_validates_saves_and_reuses(tmp_path):
    agent, market, store, search = setup_agent(tmp_path)
    result = agent.invoke("301")
    assert result.revision == 1
    assert result.relations == [relation()]
    assert len(search.calls) == 1
    assert (tmp_path / "relations/301/revisions/1.json").exists()
    assert store.load("301") == result
    # No model or API credentials are needed on cache hits, including async entry.
    reader = RelationAgent(store=store, market_cache=market)
    assert asyncio.run(reader.ainvoke("301")) == result


def test_same_profile_refresh_retains_history_and_increments_revision(tmp_path):
    agent, market, store, search = setup_agent(tmp_path)
    first = agent.invoke("301")
    model = ScriptedChatModel(responses=[
        call("search_relation_evidence", {"query": "confirm runtime relationship"}),
        call("retire_relation", {"claim_id": "power_runtime"}),
        call("finalize_relation_graph", {}), AIMessage(content="Updated"),
    ])
    updated = RelationAgent(model, store=store, market_cache=market, search=search).invoke("301", refresh=True)
    assert updated.revision == 2
    assert updated.created_at == first.created_at
    assert updated.relations[0].status == "retired"
    historical = RelationGraph.model_validate_json((tmp_path / "relations/301/revisions/1.json").read_text())
    assert historical == first


def test_profile_change_rebuilds_graph_without_silent_old_claims(tmp_path):
    agent, market, store, search = setup_agent(tmp_path)
    agent.invoke("301")
    changed = profile()
    changed.criteria[0].description = "Runtime measured under a different workload"
    market.save(market.node("301"), changed)
    model = ScriptedChatModel(responses=[
        call("search_relation_evidence", {"query": "new workload"}),
        call("upsert_open_question", {"question": {"id": "new_workload", "question": "Need comparable workload evidence"}}),
        call("finalize_relation_graph", {}), AIMessage(content="Partial graph"),
    ])
    result = RelationAgent(model, store=store, market_cache=market, search=search).invoke("301")
    assert result.revision == 2
    assert result.profile_hash == profile_hash(changed)
    assert result.relations == []
    assert result.unresolved[0].id == "new_workload"


def test_profile_fingerprint_ignores_product_sources_but_tracks_semantics():
    original = profile()
    updated = profile()
    updated.query = "different query"
    updated.item_ids = ["different_product"]
    assert profile_hash(original) == profile_hash(updated)
    updated.criteria[0].units = ["min"]
    assert profile_hash(original) != profile_hash(updated)


@pytest.mark.parametrize("mutation,match", [
    (lambda g: setattr(g.relations[0], "target", "criterion:missing"), "unknown node"),
    (lambda g: setattr(g.relations[0], "evidence_ids", ["missing"]), "unknown evidence"),
    (lambda g: g.relations.append(relation()), "duplicate IDs"),
    (lambda g: g.relations.append(relation(id="opposite", effect="positive")), "conflicting"),
    (lambda g: setattr(g, "profile_hash", "outdated"), "matching completed"),
])
def test_validation_rejects_invalid_references_and_conflicts(mutation, match):
    graph = valid_graph()
    mutation(graph)
    with pytest.raises(ValueError, match=match):
        validate_graph(graph, profile())


def test_conditions_validate_types_and_units():
    graph = valid_graph()
    graph.relations[0].conditions = [Condition(ref="attribute:power", operator="gte", value=10, unit="W")]
    validate_graph(graph, profile())
    graph.relations[0].conditions[0].unit = "kW"
    with pytest.raises(ValueError, match="declared unit"):
        validate_graph(graph, profile())
    with pytest.raises(ValidationError, match="exactly one"):
        Condition(ref="attribute:power", operator="eq", value=10, value_ref="attribute:power")


def test_utility_thresholds_and_derived_cycles():
    graph = valid_graph()
    graph.context_variables = [ContextVariable(id="required_hours", name="Required runtime", value_type="numeric", unit="h", description="Scenario need")]
    graph.utility_rules = [UtilityRule(id="runtime_value", criterion="criterion:battery_life",
        context_refs=["context:required_hours"], shape="diminishing_returns", threshold_ref="context:required_hours",
        statement="Value depends on runtime need", evidence_ids=["ev_one"], status="candidate")]
    validate_graph(graph, profile())
    graph.context_variables[0].unit = "W"
    with pytest.raises(ValueError, match="matching numeric units"):
        validate_graph(graph, profile())
    graph.utility_rules = []
    graph.derived_variables = [DerivedVariable(id="derived_power", name="Power", value_type="numeric", unit="W",
        description="derived", input_refs=["derived:derived_power"], derivation="explanation", evidence_ids=["ev_one"])]
    with pytest.raises(ValueError, match="cyclic"):
        validate_graph(graph, profile())


def test_supported_and_disputed_require_evidence():
    with pytest.raises(ValidationError, match="require evidence"):
        relation(evidence_ids=[])
    with pytest.raises(ValidationError, match="counter evidence"):
        relation(status="disputed")
    graph = valid_graph()
    graph.evidence.append(evidence("ev_two"))
    graph.relations = [relation(status="disputed", counter_evidence_ids=["ev_two"])]
    validate_graph(graph, profile())


def test_draft_edit_is_transactional_and_finalization_freezes():
    draft = RelationDraft(valid_graph(), profile())
    with pytest.raises(ValueError, match="unknown node"):
        draft.upsert("relations", relation(target="criterion:missing"))
    assert draft.graph.relations == [relation()]
    with pytest.raises(ValueError, match="search"):
        draft.finalize()
    draft.completed_searches = 1
    draft.finalize()
    with pytest.raises(ValueError, match="already finalized"):
        draft.retire("power_runtime")


def test_evidence_requires_retrieved_url_and_exact_excerpt():
    draft = RelationDraft(empty_graph(), profile())
    tools = {t.name: t for t in build_relation_tools(draft, Search())}
    args = {"id": "ev_one", "url": URL, "source_text": QUOTE, "scope": "same workload"}
    assert "error" in tools["add_evidence"].invoke(args)
    tools["search_relation_evidence"].invoke({"query": "runtime"})
    assert "error" in tools["add_evidence"].invoke(args | {"source_text": "fabricated quotation"})
    assert tools["add_evidence"].invoke(args)["saved"] == "ev_one"
    assert draft.graph.evidence[0].title == "Study"
    assert "error" in tools["add_evidence"].invoke(args | {"scope": "changed evidence"})


def test_revision_conflict_preserves_current(tmp_path):
    store = RelationStore(tmp_path)
    graph = valid_graph()
    store.save(graph, 0)
    with pytest.raises(RevisionConflict):
        store.save(graph, 0)
    assert store.load("301") == graph
    assert not (tmp_path / "301/.write.lock").exists()


def test_no_finalization_or_missing_profile_does_not_publish(tmp_path):
    agent, market, store, search = setup_agent(tmp_path, ScriptedChatModel(responses=[AIMessage(content="not finalized")]))
    with pytest.raises(RuntimeError, match="without finalizing"):
        agent.invoke("301")
    assert store.load("301") is None
    with pytest.raises(ValueError, match="persisted Market Profile"):
        agent.invoke("5322")


def test_profile_change_during_research_prevents_publication(tmp_path):
    agent, market, store, search = setup_agent(tmp_path)
    original = search.invoke
    def change_profile(args):
        changed = profile()
        changed.criteria[0].description = "changed while researching"
        market.save(market.node("301"), changed)
        return original(args)
    search.invoke = change_profile
    with pytest.raises(RevisionConflict, match="Profile changed"):
        agent.invoke("301")
    assert store.load("301") is None


def test_supervisor_exposes_relation_tool(monkeypatch):
    tools = importlib.import_module("shop_agent_langgraph.supervisor.tools")
    calls = []
    def invoke(node_id):
        calls.append(node_id)
        return valid_graph()
    monkeypatch.setattr(tools.relation_agent, "invoke", invoke)
    payload = json.loads(tools.call_relation_agent.invoke({"node_id": "301"}))
    assert payload["node_id"] == "301"
    assert calls == ["301"]
    assert tools.call_relation_agent in tools.SUPERVISOR_TOOLS


def test_pending_search_prevents_finalization():
    draft = RelationDraft(valid_graph(), profile())
    draft.completed_searches = 1
    draft.pending_searches = 1
    with pytest.raises(ValueError, match="complete evidence searches"):
        draft.finalize()


def test_failed_search_does_not_count_as_completed():
    draft = RelationDraft(valid_graph(), profile())
    class FailingSearch:
        def invoke(self, args):
            raise RuntimeError("network failure")
    tools = {t.name: t for t in build_relation_tools(draft, FailingSearch())}
    with pytest.raises(RuntimeError, match="network failure"):
        tools["search_relation_evidence"].invoke({"query": "test"})
    assert draft.pending_searches == 0
    assert draft.completed_searches == 0
    assert "error" in tools["finalize_relation_graph"].invoke({})


def test_condition_matching_and_paths_do_not_infer_missing_values_or_causality():
    from shop_agent_langgraph.agents.relation.query import ObservedValue, match_conditions, supported_paths
    graph = valid_graph()
    graph.relations[0].conditions = [Condition(ref="attribute:power", operator="gte", value=10, unit="W")]
    assert match_conditions(graph.relations[0], {}) == "unknown"
    values = {"attribute:power": ObservedValue(value=12, unit="W")}
    assert match_conditions(graph.relations[0], values) == "matched"
    assert supported_paths(graph, "attribute:power", "criterion:battery_life", values) == [["power_runtime"]]
    assert supported_paths(graph, "attribute:power", "criterion:battery_life") == []
    assert supported_paths(graph, "criterion:battery_life", "attribute:power", values) == []
    values["attribute:power"].unit = "kW"
    assert match_conditions(graph.relations[0], values) == "unknown"
    values["attribute:power"] = ObservedValue(value=2, unit="W")
    assert match_conditions(graph.relations[0], values) == "unmatched"
    graph.relations[0].status = "candidate"
    assert supported_paths(graph, "attribute:power", "criterion:battery_life", values) == []


def test_search_budget_returns_actionable_error():
    draft = RelationDraft(empty_graph(), profile())
    draft.search_count = 12
    search = Search()
    tools = {t.name: t for t in build_relation_tools(draft, search)}
    assert "budget exhausted" in tools["search_relation_evidence"].invoke({"query": "test"})["error"]
    assert search.calls == []
