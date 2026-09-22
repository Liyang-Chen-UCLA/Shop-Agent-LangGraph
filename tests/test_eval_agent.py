import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from shop_agent_langgraph.agents.eval.graph import EvalAgent
from shop_agent_langgraph.agents.eval.schemas import PartialResolutionBatch
from shop_agent_langgraph.agents.eval.tools import build_eval_tools
from shop_agent_langgraph.domain.criteria import BooleanAttribute, CriteriaAttributeSet


def item(item_id: str, description: str | None = None) -> dict[str, object]:
    return {
        "id": item_id,
        "name": item_id,
        "description": description or item_id,
        "aliases": [],
        "type": "boolean",
    }


def collections() -> tuple[CriteriaAttributeSet, CriteriaAttributeSet]:
    left = CriteriaAttributeSet(
        source_item_ids=["product_1"],
        criteria=[],
        attributes=[
            BooleanAttribute(**item("vibration_bundle", "Has adjustable vibration")),
            BooleanAttribute(**item("lighting", "Has decorative lighting")),
        ],
    )
    right = CriteriaAttributeSet(
        source_item_ids=["product_2", "product_1"],
        criteria=[],
        attributes=[
            BooleanAttribute(**item("rumble", "Has vibration feedback")),
            BooleanAttribute(
                **item("dynamic_light_bar", "Has a dynamic decorative light bar")
            ),
        ],
    )
    return left, right


L1, L2 = "left:attribute:vibration_bundle", "left:attribute:lighting"
R1, R2 = "right:attribute:rumble", "right:attribute:dynamic_light_bar"


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


def group() -> dict[str, object]:
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


def setup_tools():
    tools, unresolved, criteria, attributes = build_eval_tools(*collections())
    return {tool.name: tool for tool in tools}, unresolved, criteria, attributes


def test_many_to_many_split_and_runtime_submission() -> None:
    tools, unresolved, criteria, attributes = setup_tools()
    assert "get_diff" not in tools
    result = tools["resolve_partial"].invoke({"resolutions": [group()]})
    assert result["committed"] is True
    assert not unresolved
    assert len(attributes) == 4 and not criteria
    assert result["results"][0]["outputs"][1]["source_refs"] == [L1]
    final = tools["submit"].invoke({})
    assert final["source_item_ids"] == ["product_1", "product_2"]
    assert {value["id"] for value in final["attributes"]} == {
        "has_vibration",
        "vibration_adjustable",
        "has_lighting",
        "has_dynamic_light_bar",
    }
    with pytest.raises(ValidationError):
        tools["submit"].get_input_schema().model_validate(
            {"source_item_ids": ["forged"], "criteria": [], "attributes": []}
        )


def split_groups() -> tuple[dict[str, object], dict[str, object]]:
    first, second = group(), group()
    first.update(
        group_id="rumble",
        left_ids=[L1],
        right_ids=[R1],
        outputs=first["outputs"][:2],
    )
    second.update(
        group_id="lighting",
        left_ids=[L2],
        right_ids=[R2],
        outputs=second["outputs"][2:],
    )
    return first, second


def test_multiple_groups_in_one_batch() -> None:
    tools, unresolved, _, attributes = setup_tools()
    result = tools["resolve_partial"].invoke({"resolutions": list(split_groups())})
    assert result["committed"] and len(result["results"]) == 2
    assert not unresolved and len(attributes) == 4


@pytest.mark.parametrize(
    "problem",
    [
        "unknown_ref",
        "wrong_side",
        "uncovered_source",
        "foreign_output_ref",
        "duplicate_output_ref",
        "match_single_side",
        "independent_both_sides",
        "duplicate_output_id",
        "blank_aspect",
        "overlapping_groups",
        "duplicate_group_id",
    ],
)
def test_invalid_partial_batch_is_atomic(problem: str) -> None:
    tools, unresolved, criteria, attributes = setup_tools()
    before = dict(unresolved)
    first, second = split_groups()
    if problem == "unknown_ref":
        second["right_ids"] = ["right:attribute:missing"]
    if problem == "wrong_side":
        second["left_ids"] = [R2]
    if problem == "uncovered_source":
        second["outputs"] = [output("only_left", [L2], "independent")]
    if problem == "foreign_output_ref":
        second["outputs"][0]["source_refs"].append(L1)
    if problem == "duplicate_output_ref":
        second["outputs"][0]["source_refs"].append(L2)
    if problem == "match_single_side":
        second["outputs"][0]["source_refs"] = [L2]
    if problem == "independent_both_sides":
        second["outputs"][0]["alignment"] = "independent"
    if problem == "duplicate_output_id":
        second["outputs"][0]["item"]["id"] = "has_vibration"
    if problem == "blank_aspect":
        second["outputs"][0]["preserved_aspects"] = [" "]
    if problem == "overlapping_groups":
        second["left_ids"].append(L1)
        second["outputs"][0]["source_refs"].append(L1)
    if problem == "duplicate_group_id":
        second["group_id"] = first["group_id"]
    result = tools["resolve_partial"].invoke({"resolutions": [first, second]})
    assert result["committed"] is False
    assert unresolved == before and not criteria and not attributes


def test_needs_review_retains_inputs_and_blocks_submit() -> None:
    tools, unresolved, _, attributes = setup_tools()
    first, second = split_groups()
    second.update(
        status="needs_review",
        outputs=[],
        missing_evidence=["Which lighting controls exist?"],
    )
    result = tools["resolve_partial"].invoke({"resolutions": [first, second]})
    assert result["committed"] is True
    assert result["results"][1]["consumed_refs"] == []
    assert set(unresolved) == {L2, R2} and len(attributes) == 2
    with pytest.raises(ValueError, match="unresolved"):
        tools["submit"].invoke({})
    result = tools["resolve_partial"].invoke(
        {"resolutions": [split_groups()[1]]}
    )
    assert result["committed"] and not unresolved


def test_partial_can_retain_all_dimensions_as_independent() -> None:
    tools, unresolved, _, _ = setup_tools()
    resolution = group()
    resolution["outputs"] = [
        output(f"dimension_{index}", [ref], "independent")
        for index, ref in enumerate([L1, L2, R1, R2])
    ]
    assert tools["resolve_partial"].invoke(
        {"resolutions": [resolution]}
    )["committed"]
    assert not unresolved


@pytest.mark.parametrize("problem", ["kind_mismatch", "empty_batch", "invalid_review"])
def test_schema_rejects_invalid_payload(problem: str) -> None:
    payload = {"resolutions": [group()]}
    if problem == "kind_mismatch":
        payload["resolutions"][0]["outputs"][0]["kind"] = "criterion"
    if problem == "empty_batch":
        payload["resolutions"] = []
    if problem == "invalid_review":
        payload["resolutions"][0]["status"] = "needs_review"
    with pytest.raises(ValidationError):
        PartialResolutionBatch.model_validate(payload)


@pytest.mark.parametrize("operation", ["match", "independent"])
def test_existing_operations_do_not_partially_consume_invalid_batch(
    operation: str,
) -> None:
    tools, unresolved, criteria, attributes = setup_tools()
    before = dict(unresolved)
    if operation == "match":
        args = {
            "matches": [
                {
                    "left_id": L1,
                    "right_id": "right:attribute:missing",
                    "kind": "attribute",
                    "item": item("merged"),
                }
            ]
        }
    else:
        args = {"item_ids": [L1, "right:attribute:missing"]}
    with pytest.raises(ValueError):
        tools[operation].invoke(args)
    assert unresolved == before and not criteria and not attributes


def test_match_rejects_reversed_sources_without_mutation() -> None:
    tools, unresolved, criteria, attributes = setup_tools()
    before = dict(unresolved)
    with pytest.raises(ValueError, match="expected a left source"):
        tools["match"].invoke(
            {
                "matches": [
                    {
                        "left_id": R1,
                        "right_id": L1,
                        "kind": "attribute",
                        "item": item("merged"),
                    }
                ]
            }
        )
    assert unresolved == before and not criteria and not attributes


def test_match_rejects_conflicting_output_ids_without_mutation() -> None:
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


def test_duplicate_inputs_are_not_silently_overwritten() -> None:
    left, right = collections()
    left.attributes.append(left.attributes[0])
    with pytest.raises(ValueError, match="duplicate input"):
        build_eval_tools(left, right)


class ScriptedModel:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.messages_seen = []

    def bind_tools(self, tools):
        self.tools = {tool.name for tool in tools}
        return self

    def invoke(self, messages):
        self.messages_seen.append(messages)
        return AIMessage(content="", tool_calls=next(self.calls))


def call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_graph_executes_finalize_tool_and_rejects_forged_submission() -> None:
    model = ScriptedModel(
        [
            [call("resolve_partial", {"resolutions": [group()]}, "repair")],
            [
                call(
                    "submit",
                    {
                        "source_item_ids": ["forged"],
                        "criteria": [],
                        "attributes": [],
                    },
                    "bad",
                )
            ],
            [call("submit", {}, "final")],
        ]
    )
    result = EvalAgent(model).invoke(*collections())
    assert result.source_item_ids == ["product_1", "product_2"]
    assert len(result.attributes) == 4
    assert any("rejected" in str(message.content) for message in model.messages_seen[-1])


def test_graph_rejects_finalize_mixed_with_mutations_before_any_write() -> None:
    model = ScriptedModel(
        [
            [
                call(
                    "resolve_partial",
                    {"resolutions": [group()]},
                    "repair_bad",
                ),
                call("submit", {}, "mixed"),
            ],
            [call("resolve_partial", {"resolutions": [group()]}, "repair")],
            [call("submit", {}, "final")],
        ]
    )
    result = EvalAgent(model).invoke(*collections())
    assert len(result.attributes) == 4
    assert any(
        "No tools executed" in str(message.content)
        for message in model.messages_seen[1]
    )


@pytest.mark.parametrize("left_count,right_count", [(1, 2), (2, 1), (2, 2)])
def test_variable_group_cardinality(left_count: int, right_count: int) -> None:
    tools, unresolved, _, _ = setup_tools()
    resolution = group()
    resolution["left_ids"] = [L1, L2][:left_count]
    resolution["right_ids"] = [R1, R2][:right_count]
    refs = resolution["left_ids"] + resolution["right_ids"]
    resolution["outputs"] = [
        output(f"retained_{index}", [ref], "independent")
        for index, ref in enumerate(refs)
    ]
    result = tools["resolve_partial"].invoke({"resolutions": [resolution]})
    assert result["committed"]
    assert set(unresolved) == {L1, L2, R1, R2} - set(refs)


def test_non_eval_submission_mode_remains_compatible() -> None:
    from shop_agent_langgraph.agents.research.schemas import ResearchResult
    from shop_agent_langgraph.agents.research.tools import submit_research_result
    from shop_agent_langgraph.core.submit_agent import build_submit_agent_graph

    model = ScriptedModel(
        [
            [
                call(
                    "submit_research_result",
                    {"item_id": "product_1", "criteria": [], "attributes": []},
                    "research",
                )
            ]
        ]
    )
    graph = build_submit_agent_graph(
        model=model,
        tools=[submit_research_result],
        system_prompt="Test",
        result_schema=ResearchResult,
        submit_tool_name="submit_research_result",
        name="test",
    )
    result = graph.invoke({"messages": [{"role": "user", "content": "test"}]})
    assert result["submitted_result"].item_id == "product_1"
