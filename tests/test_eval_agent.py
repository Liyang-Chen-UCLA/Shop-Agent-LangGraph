import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError

from shop_agent_langgraph.agents.eval.graph import EvalAgent, _validate_submission
from shop_agent_langgraph.agents.eval.schemas import MatchDecision
from shop_agent_langgraph.agents.eval.tools import build_eval_tools
from shop_agent_langgraph.domain.criteria import (
    BooleanAttribute,
    BooleanCriterion,
    CriteriaAttributeSet,
    TrueBetter,
)


def criterion(item_id: str) -> BooleanCriterion:
    return BooleanCriterion(
        id=item_id,
        name=item_id,
        description=item_id,
        aliases=[],
        type="boolean",
        direction=TrueBetter(type="true_better"),
    )


def attribute(item_id: str) -> BooleanAttribute:
    return BooleanAttribute(
        id=item_id,
        name=item_id,
        description=item_id,
        aliases=[],
        type="boolean",
    )


LEFT = CriteriaAttributeSet(
    source_item_ids=["left-product", "shared-product"],
    criteria=[criterion("battery_life")],
    attributes=[],
)
RIGHT = CriteriaAttributeSet(
    source_item_ids=["shared-product", "right-product"],
    criteria=[],
    attributes=[attribute("has_bluetooth")],
)


class SequencedChatModel(BaseChatModel):
    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "sequenced-test-model"

    def bind_tools(self, tools: object, **kwargs: object) -> BaseChatModel:
        return self

    def _generate(self, messages: object, **kwargs: object) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=self.responses.pop(0))]
        )


def validate(
    result: CriteriaAttributeSet,
    *,
    criteria: list[BooleanCriterion] | None = None,
    attributes: list[BooleanAttribute] | None = None,
) -> None:
    _validate_submission(
        result,
        left=LEFT,
        right=RIGHT,
        unresolved={},
        criteria=criteria if criteria is not None else LEFT.criteria,
        attributes=attributes if attributes is not None else RIGHT.attributes,
    )


def eval_tool(
    left: CriteriaAttributeSet,
    right: CriteriaAttributeSet,
    name: str,
) -> tuple[object, dict[str, object], list[object], list[object]]:
    tools, unresolved, criteria, attributes = build_eval_tools(left, right)
    selected = next(tool for tool in tools if tool.name == name)
    return selected, unresolved, criteria, attributes


def test_submission_accepts_accumulated_result_in_any_order() -> None:
    second = criterion("price")
    result = CriteriaAttributeSet(
        source_item_ids=["right-product", "left-product", "shared-product"],
        criteria=[second, LEFT.criteria[0]],
        attributes=RIGHT.attributes,
    )

    validate(result, criteria=[LEFT.criteria[0], second])


def test_submission_rejects_polluted_source_ids() -> None:
    result = CriteriaAttributeSet(
        source_item_ids=["wrong-product"],
        criteria=LEFT.criteria,
        attributes=RIGHT.attributes,
    )

    with pytest.raises(ValueError, match="deduplicated union"):
        validate(result)


def test_submission_rejects_unprocessed_item() -> None:
    result = CriteriaAttributeSet(
        source_item_ids=["left-product", "shared-product", "right-product"],
        criteria=[*LEFT.criteria, criterion("invented")],
        attributes=RIGHT.attributes,
    )

    with pytest.raises(ValueError, match="criteria do not match"):
        validate(result)


def test_submission_rejects_duplicate_canonical_ids_across_kinds() -> None:
    duplicate_attribute = attribute("battery_life")
    result = CriteriaAttributeSet(
        source_item_ids=["left-product", "shared-product", "right-product"],
        criteria=LEFT.criteria,
        attributes=[duplicate_attribute],
    )

    with pytest.raises(ValueError, match="canonical item IDs must be unique"):
        validate(result, attributes=[duplicate_attribute])


def test_eval_graph_retries_after_rejecting_polluted_submit_arguments() -> None:
    empty_left = CriteriaAttributeSet(
        source_item_ids=["left-product"], criteria=[], attributes=[]
    )
    empty_right = CriteriaAttributeSet(
        source_item_ids=["right-product"], criteria=[], attributes=[]
    )
    model = SequencedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit",
                        "id": "bad-submit",
                        "args": {
                            "source_item_ids": ["polluted-product"],
                            "criteria": [],
                            "attributes": [],
                        },
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit",
                        "id": "good-submit",
                        "args": {
                            "source_item_ids": ["right-product", "left-product"],
                            "criteria": [],
                            "attributes": [],
                        },
                    }
                ],
            ),
        ]
    )

    result = EvalAgent(model).invoke(empty_left, empty_right)

    assert set(result.source_item_ids) == {"left-product", "right-product"}
    assert model.responses == []


def test_match_batch_failure_is_atomic() -> None:
    left = CriteriaAttributeSet(
        source_item_ids=["left-product"],
        criteria=[criterion("first"), criterion("second")],
        attributes=[],
    )
    right = CriteriaAttributeSet(
        source_item_ids=["right-product"],
        criteria=[criterion("first"), criterion("second")],
        attributes=[],
    )
    match, unresolved, criteria, attributes = eval_tool(left, right, "match")
    original_unresolved = dict(unresolved)

    with pytest.raises(ValueError, match="do not exist"):
        match.invoke(  # type: ignore[attr-defined]
            {
                "matches": [
                    {
                        "left_id": "left:criterion:first",
                        "right_id": "right:criterion:first",
                        "kind": "criterion",
                        "item": criterion("first").model_dump(),
                    },
                    {
                        "left_id": "left:criterion:second",
                        "right_id": "right:criterion:missing",
                        "kind": "criterion",
                        "item": criterion("second").model_dump(),
                    },
                ]
            }
        )

    assert unresolved == original_unresolved
    assert criteria == []
    assert attributes == []


def test_independent_batch_failure_is_atomic() -> None:
    independent, unresolved, criteria, attributes = eval_tool(
        LEFT, RIGHT, "independent"
    )
    original_unresolved = dict(unresolved)

    with pytest.raises(ValueError, match="do not exist"):
        independent.invoke(  # type: ignore[attr-defined]
            {
                "item_ids": [
                    "left:criterion:battery_life",
                    "right:attribute:missing",
                ]
            }
        )

    assert unresolved == original_unresolved
    assert criteria == []
    assert attributes == []


def test_match_rejects_reused_reference_without_mutation() -> None:
    match, unresolved, criteria, attributes = eval_tool(LEFT, RIGHT, "match")
    original_unresolved = dict(unresolved)

    with pytest.raises(ValueError, match="references must be unique"):
        match.invoke(  # type: ignore[attr-defined]
            {
                "matches": [
                    {
                        "left_id": "left:criterion:battery_life",
                        "right_id": "left:criterion:battery_life",
                        "kind": "criterion",
                        "item": criterion("battery_life").model_dump(),
                    }
                ]
            }
        )

    assert unresolved == original_unresolved
    assert criteria == []
    assert attributes == []


def test_match_rejects_reversed_source_sides_without_mutation() -> None:
    match, unresolved, criteria, attributes = eval_tool(LEFT, RIGHT, "match")
    original_unresolved = dict(unresolved)

    with pytest.raises(ValueError, match="left reference followed by a right"):
        match.invoke(  # type: ignore[attr-defined]
            {
                "matches": [
                    {
                        "left_id": "right:attribute:has_bluetooth",
                        "right_id": "left:criterion:battery_life",
                        "kind": "criterion",
                        "item": criterion("connectivity").model_dump(),
                    }
                ]
            }
        )

    assert unresolved == original_unresolved
    assert criteria == []
    assert attributes == []


def test_match_decision_rejects_kind_item_type_mismatch() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        MatchDecision(
            left_id="left:attribute:has_bluetooth",
            right_id="right:attribute:has_bluetooth",
            kind="criterion",
            item=attribute("has_bluetooth"),
        )


def test_build_eval_tools_rejects_duplicate_input_ids() -> None:
    duplicate_input = CriteriaAttributeSet(
        source_item_ids=["left-product"],
        criteria=[criterion("duplicate"), criterion("duplicate")],
        attributes=[],
    )

    with pytest.raises(ValueError, match="duplicate item IDs in left input"):
        build_eval_tools(duplicate_input, RIGHT)


def test_match_rejects_conflicting_batch_output_ids_without_mutation() -> None:
    left = CriteriaAttributeSet(
        source_item_ids=["left-product"],
        criteria=[criterion("first"), criterion("second")],
        attributes=[],
    )
    right = CriteriaAttributeSet(
        source_item_ids=["right-product"],
        criteria=[criterion("first"), criterion("second")],
        attributes=[],
    )
    match, unresolved, criteria, attributes = eval_tool(left, right, "match")
    original_unresolved = dict(unresolved)

    with pytest.raises(ValueError, match="output IDs would conflict"):
        match.invoke(  # type: ignore[attr-defined]
            {
                "matches": [
                    {
                        "left_id": "left:criterion:first",
                        "right_id": "right:criterion:first",
                        "kind": "criterion",
                        "item": criterion("same_output").model_dump(),
                    },
                    {
                        "left_id": "left:criterion:second",
                        "right_id": "right:criterion:second",
                        "kind": "criterion",
                        "item": criterion("same_output").model_dump(),
                    },
                ]
            }
        )

    assert unresolved == original_unresolved
    assert criteria == []
    assert attributes == []
