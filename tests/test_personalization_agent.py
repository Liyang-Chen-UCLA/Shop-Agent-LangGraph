import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import ValidationError

from shop_agent_langgraph.agents.personalization import PersonalizationAgent
from shop_agent_langgraph.agents.personalization.schemas import (
    ClaimReferenceError, OutputValidationError, PersonalizationContent,
    QuestionSet, validate_output,
)
from shop_agent_langgraph.agents.personalization.store import PersonalizationStore
from shop_agent_langgraph.agents.relation.schemas import ContextVariable
from shop_agent_langgraph.agents.relation.store import RelationStore, RevisionConflict
from shop_agent_langgraph.domain.market_cache import MarketCache
from shop_agent_langgraph.supervisor.tools import _personalization_history, prepare_personalization
from test_relation_agent import ScriptedChatModel, call, profile, valid_graph


def questions():
    return {"questions": [{"id": "usage", "text": "每天使用多久？",
        "target_refs": ["context:usage_hours"], "claim_ids": [],
        "purpose": "确认续航需求"}]}


def content():
    return {"context": [{"ref": "context:usage_hours", "value": 4, "unit": "h",
                         "user_quote": "每天四小时"}],
            "criteria": [{"source_ref": "criterion:battery_life", "preference": "续航优先",
                "priority": "high", "strength": "soft", "user_quote": "续航优先",
                "claim_ids": ["power_runtime"]}], "attributes": [], "unresolved": []}


def setup(tmp_path, outputs=None):
    market = MarketCache(tmp_path / "market")
    market.save(market.node("301"), profile())
    relations = RelationStore(tmp_path / "relations")
    graph = valid_graph()
    graph.context_variables = [ContextVariable(id="usage_hours", name="Usage",
        value_type="numeric", unit="h", description="Daily usage")]
    relations.save(graph, 0)
    model = ScriptedChatModel(responses=outputs or [
        call("QuestionSet", questions()), call("PersonalizationContent", content())])
    store = PersonalizationStore(tmp_path / "personalization")
    return PersonalizationAgent(model, market_cache=market, relation_store=relations,
                                store=store), model, graph


def test_two_stages_save_and_update_without_mutating_shared_inputs(tmp_path):
    replacement = {"context": [], "criteria": [], "attributes": [], "unresolved": []}
    agent, model, graph = setup(tmp_path, [call("QuestionSet", questions()),
        call("PersonalizationContent", content()), call("PersonalizationContent", replacement)])
    history = ["想买手柄"]
    assert len(agent.prepare_questions("301", history, thread_id="a", task_id="t").questions) == 1
    history.append("每天四小时，续航优先")
    result = agent.personalize("301", history, thread_id="a", task_id="t")
    assert result.graph_revision == 1
    assert result.criteria[0].priority == "high"
    assert agent.store.load("a", "t").result == result
    history.append("删掉之前的全部偏好与场景信息")
    updated = asyncio.run(agent.apersonalize("301", history, thread_id="a", task_id="t"))
    assert updated.criteria == []
    assert agent.store.load("a", "t").revision == 3
    assert agent.relation_store.load("301") == graph
    assert agent.market_cache.load(agent.market_cache.node("301")) == profile()
    assert agent.store.load("b", "t") is None
    assert agent.store.load("a", "other") is None


def test_empty_questions_can_finalize_without_feedback(tmp_path):
    agent, _, _ = setup(tmp_path, [call("QuestionSet", {"questions": []}),
                                  call("PersonalizationContent", content())])
    history = ["每天四小时，续航优先"]
    asyncio.run(agent.aprepare_questions("301", history, thread_id="a", task_id="t"))
    assert agent.personalize("301", history, thread_id="a", task_id="t").criteria


def test_feedback_is_required_and_history_cannot_be_replaced(tmp_path):
    agent, model, _ = setup(tmp_path)
    agent.prepare_questions("301", ["买手柄"], thread_id="a", task_id="t")
    with pytest.raises(ValueError, match="wait for user"):
        agent.personalize("301", ["买手柄"], thread_id="a", task_id="t")
    with pytest.raises(ValueError, match="full chronological"):
        agent.personalize("301", ["invented history"], thread_id="a", task_id="t")
    assert model.position == 1


@pytest.mark.parametrize("failure", ["quote", "reference", "claim", "unit", "type", "duplicate"])
def test_invalid_results_do_not_replace_saved_session(tmp_path, failure):
    bad = content()
    if failure == "quote":
        bad["criteria"][0]["user_quote"] = "用户从未说过"
    elif failure == "reference":
        bad["criteria"][0]["source_ref"] = "criterion:missing"
    elif failure == "claim":
        bad["criteria"][0]["claim_ids"] = ["missing"]
    elif failure == "unit":
        bad["context"][0]["unit"] = "W"
    elif failure == "type":
        bad["context"][0]["value"] = True
    else:
        bad["criteria"].append(dict(bad["criteria"][0]))
    agent, _, _ = setup(tmp_path, [call("QuestionSet", questions())] +
        [call("PersonalizationContent", bad) for _ in range(3)])
    agent.prepare_questions("301", ["买手柄"], thread_id="a", task_id="t")
    before = agent.store.load("a", "t")
    with pytest.raises(ValueError):
        agent.personalize("301", ["买手柄", "每天四小时，续航优先"], thread_id="a", task_id="t")
    assert agent.store.load("a", "t") == before


def test_reject_stale_graph_and_missing_inputs_before_model_call(tmp_path):
    agent, model, graph = setup(tmp_path)
    agent.prepare_questions("301", ["买手柄"], thread_id="a", task_id="t")
    agent.relation_store.save(graph.model_copy(update={"revision": 2}), 1)
    with pytest.raises(RevisionConflict, match="upstream"):
        agent.personalize("301", ["买手柄", "每天四小时"], thread_id="a", task_id="t")
    with pytest.raises(ValueError, match="persisted"):
        agent.prepare_questions("302", ["买东西"], thread_id="a", task_id="new")
    assert model.position == 1


def test_question_limits_references_and_claim_status(tmp_path):
    with pytest.raises(ValidationError):
        QuestionSet(questions=questions()["questions"] * 4)
    agent, _, graph = setup(tmp_path)
    q = QuestionSet.model_validate(questions())
    q.questions[0].target_refs = ["context:missing"]
    with pytest.raises(ValueError, match="unknown"):
        validate_output(q, profile(), graph, ["hi"])
    graph.relations[0].status = "candidate"
    with pytest.raises(ValueError, match="supported"):
        validate_output(PersonalizationContent.model_validate(content()), profile(), graph,
                        ["每天四小时，续航优先"])


def test_store_conflicts_and_safe_identity_paths(tmp_path):
    agent, _, _ = setup(tmp_path)
    agent.prepare_questions("301", ["买手柄"], thread_id="../../a", task_id="../t")
    saved = agent.store.load("../../a", "../t")
    assert agent.store.path("../../a", "../t").is_relative_to(agent.store.directory)
    with pytest.raises(RevisionConflict):
        agent.store.save(saved.model_copy(update={"revision": 2}), 0)
    assert agent.store.load("../../a", "../t") == saved


def test_runtime_identity_and_messages_are_not_model_arguments():
    schema = prepare_personalization.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"node_id"}
    runtime = SimpleNamespace(config={"configurable": {"thread_id": "a"}}, state={"messages": [
        HumanMessage(content="old task", id="old"), HumanMessage(content="new task", id="new"),
        HumanMessage(content="feedback", id="reply")]})
    assert _personalization_history(runtime, "new") == ("a", "new", ["new task", "feedback"])
    with pytest.raises(ValueError, match="not present"):
        _personalization_history(runtime, "other-thread")
    runtime.state["messages"].append(ToolMessage(
        content='{"task_id":"new"}', name="prepare_personalization", tool_call_id="p"))
    with pytest.raises(ValueError, match="latest prepared"):
        _personalization_history(runtime, "old")


@pytest.mark.parametrize("block_content", [False, True])
def test_supervisor_tool_roundtrip_passes_thread_and_real_feedback(tmp_path, monkeypatch, block_content):
    from langgraph.checkpoint.memory import InMemorySaver
    from shop_agent_langgraph.supervisor.graph import Supervisor, build_supervisor_graph
    import importlib
    tools_module = importlib.import_module("shop_agent_langgraph.supervisor.tools")
    from langchain_core.messages import AIMessage

    agent, _, _ = setup(tmp_path)
    monkeypatch.setattr(tools_module, "personalization_agent", agent)
    model = ScriptedChatModel(responses=[
        call("prepare_personalization", {"node_id": "301"}), AIMessage(content="每天使用多久？")])
    supervisor = Supervisor(build_supervisor_graph(model, checkpointer=InMemorySaver()))
    def invoke(text):
        if not block_content:
            return supervisor.invoke(text, thread_id="actual-thread")
        state = supervisor.graph.invoke({"messages": [{"type": "human", "content": [
            {"type": "text", "text": text}]}]},
            config={"configurable": {"thread_id": "actual-thread"}})
        return state["messages"][-1].content

    assert invoke("买手柄") == "每天使用多久？"
    task_id = supervisor.get_state("actual-thread")["messages"][0].id
    assert agent.store.load("actual-thread", task_id).user_messages == ["买手柄"]
    model.responses.extend([call("complete_personalization", {"node_id": "301", "task_id": task_id}),
                            AIMessage(content="已保存偏好")])
    assert invoke("每天四小时，续航优先") == "已保存偏好"
    assert agent.store.load("actual-thread", task_id).result.criteria[0].user_quote == "续航优先"


@pytest.mark.parametrize("content, expected", [
    ([{"text": "狗粮", "type": "text"}], "狗粮"),
    (["  狗粮", {"type": "text", "text": "预算 200  "}], "  狗粮\n预算 200  "),
])
def test_history_normalizes_text_blocks(content, expected):
    runtime = SimpleNamespace(config={"configurable": {"thread_id": "trace-thread"}},
        state={"messages": [HumanMessage(content=content, id="trace-message")]})
    assert _personalization_history(runtime) == ("trace-thread", "trace-message", [expected])


@pytest.mark.parametrize("block", [
    {"type": "image_url", "image_url": {"url": "https://example.org/image.png"}},
    {"type": "text", "text": 123},
])
def test_history_does_not_turn_non_text_blocks_into_user_quotes(block):
    runtime = SimpleNamespace(config={"configurable": {"thread_id": "a"}},
        state={"messages": [HumanMessage(content=[block], id="m")]})
    with pytest.raises(ValueError, match="non-text user content blocks"):
        _personalization_history(runtime)


class CapturingPersonalizationModel(ScriptedChatModel):
    payloads: list[dict] = []

    def _generate(self, messages, **kwargs):
        self.payloads.append(json.loads(messages[-1].content))
        return super()._generate(messages, **kwargs)


@pytest.mark.parametrize("stage", ["questions", "profile"])
def test_invalid_claim_is_returned_to_model_for_correction(tmp_path, stage):
    bad = questions() if stage == "questions" else content()
    entry = bad["questions"][0] if stage == "questions" else bad["criteria"][0]
    entry["claim_ids"] = ["invented_claim"]
    schema = "QuestionSet" if stage == "questions" else "PersonalizationContent"
    responses = [] if stage == "questions" else [call("QuestionSet", questions())]
    responses += [call(schema, bad), call(schema, questions() if stage == "questions" else content())]
    agent, _, _ = setup(tmp_path)
    model = CapturingPersonalizationModel(responses=responses)
    agent.model = model
    history = ["买手柄"]
    agent.prepare_questions("301", history, thread_id="a", task_id="t")
    if stage == "profile":
        history.append("每天四小时，续航优先")
        result = agent.personalize("301", history, thread_id="a", task_id="t")
        assert result.criteria[0].claim_ids == ["power_runtime"]
    correction = model.payloads[-1]
    assert correction["allowed_claim_ids"] == ["power_runtime"]
    assert "invented_claim" in correction["validation_feedback"]["error"]
    assert "unknown ID" in correction["validation_feedback"]["error"]
    assert correction["validation_feedback"]["rejected_output"] == bad
    assert correction["user_messages"] == history
    assert agent.store.load("a", "t").user_messages == history


def test_duplicate_source_ref_is_returned_to_model_for_correction(tmp_path):
    bad = content()
    bad["criteria"].append(dict(bad["criteria"][0]))
    agent, _, _ = setup(tmp_path)
    model = CapturingPersonalizationModel(responses=[
        call("QuestionSet", questions()),
        call("PersonalizationContent", bad),
        call("PersonalizationContent", content()),
    ])
    agent.model = model
    history = ["买手柄"]
    agent.prepare_questions("301", history, thread_id="a", task_id="t")
    history.append("每天四小时，续航优先")

    result = agent.personalize("301", history, thread_id="a", task_id="t")

    assert len(result.criteria) == 1
    correction = model.payloads[-1]
    assert "criterion:battery_life" in correction["validation_feedback"]["error"]
    assert "source_ref at most once" in correction["validation_feedback"]["instruction"]


def test_exhausted_duplicate_corrections_do_not_publish(tmp_path):
    bad = content()
    bad["criteria"].append(dict(bad["criteria"][0]))
    agent, model, _ = setup(tmp_path, [call("QuestionSet", questions())] +
        [call("PersonalizationContent", bad) for _ in range(3)])
    history = ["买手柄"]
    agent.prepare_questions("301", history, thread_id="a", task_id="t")
    before = agent.store.load("a", "t")

    with pytest.raises(
        OutputValidationError,
        match="after 3 attempts.*criterion:battery_life",
    ):
        agent.personalize(
            "301", history + ["每天四小时，续航优先"], thread_id="a", task_id="t"
        )

    assert model.position == 4
    assert agent.store.load("a", "t") == before


def test_exhausted_question_claim_corrections_do_not_publish(tmp_path):
    bad = questions()
    bad["questions"][0]["claim_ids"] = ["invented_claim"]
    agent, model, _ = setup(tmp_path, [call("QuestionSet", bad) for _ in range(3)])
    with pytest.raises(ClaimReferenceError, match="after 3 attempts.*invented_claim"):
        agent.prepare_questions("301", ["买手柄"], thread_id="a", task_id="t")
    assert model.position == 3
    assert agent.store.load("a", "t") is None


@pytest.mark.parametrize("status", ["candidate", "disputed", "retired"])
def test_claim_error_identifies_non_supported_status(status):
    graph = valid_graph()
    graph.relations[0].status = status
    q = questions()
    q["questions"][0]["target_refs"] = ["criterion:battery_life"]
    q["questions"][0]["claim_ids"] = ["power_runtime"]
    with pytest.raises(ClaimReferenceError, match=f"question usage.*power_runtime.*{status}"):
        validate_output(QuestionSet.model_validate(q), profile(), graph, ["买手柄"])
