import asyncio

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from shop_agent_langgraph.agents.eval.graph import EvalAgent
from shop_agent_langgraph.agents.eval.schemas import EvalGroup, EvalReport
from shop_agent_langgraph.agents.eval.tools import validate_eval_report
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
                attribute("vibration", "Has vibration"),
                attribute("adjustable_vibration", "Vibration intensity is adjustable"),
            ],
        ),
        CriteriaAttributeSet(
            source_item_ids=["product_2"],
            criteria=[],
            attributes=[
                attribute("rumble", "Has rumble feedback"),
                attribute("haptics", "Composite haptic feedback support"),
            ],
        ),
    )


L1 = "left:attribute:vibration"
L2 = "left:attribute:adjustable_vibration"
R1 = "right:attribute:rumble"
R2 = "right:attribute:haptics"


def report_payload() -> dict[str, object]:
    return {
        "groups": [
            {
                "status": "match",
                "left_ids": [L1],
                "right_ids": [R1],
                "reason": "Both describe vibration feedback.",
            },
            {
                "status": "uncertain",
                "left_ids": [L2],
                "right_ids": [R2],
                "reason": "Adjustability may be one aspect of composite haptics.",
            },
        ]
    }


def call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class ScriptedModel:
    def __init__(self, calls: list[list[dict[str, object]]]) -> None:
        self.calls = iter(calls)
        self.messages_seen: list[object] = []
        self.tool_names: set[str] = set()

    def bind_tools(self, tools):
        self.tool_names = {tool.name for tool in tools}
        return self

    def invoke(self, messages):
        self.messages_seen.append(messages)
        return AIMessage(content="", tool_calls=next(self.calls))


def test_eval_returns_report_without_modifying_inputs() -> None:
    left, right = collections()
    left_before = left.model_dump(mode="json")
    right_before = right.model_dump(mode="json")
    model = ScriptedModel(
        [[call("submit_eval_report", report_payload(), "submit-report")]]
    )

    result = EvalAgent(model).invoke(left, right)  # type: ignore[arg-type]

    assert isinstance(result, EvalReport)
    assert [group.status for group in result.groups] == ["match", "uncertain"]
    assert left.model_dump(mode="json") == left_before
    assert right.model_dump(mode="json") == right_before
    assert model.tool_names == {"submit_eval_report"}


def test_eval_ainvoke_returns_eval_report() -> None:
    model = ScriptedModel(
        [[call("submit_eval_report", report_payload(), "submit-report")]]
    )
    result = asyncio.run(EvalAgent(model).ainvoke(*collections()))  # type: ignore[arg-type]
    assert isinstance(result, EvalReport)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_side", "unknown"])
def test_eval_report_requires_exact_source_coverage(problem: str) -> None:
    left, right = collections()
    payload = report_payload()
    groups = payload["groups"]
    if problem == "missing":
        groups[1]["left_ids"] = []
        groups[1]["status"] = "independent"
    elif problem == "duplicate":
        groups[1]["left_ids"] = [L1, L2]
    elif problem == "wrong_side":
        groups[0]["left_ids"] = [R1]
        groups[0]["right_ids"] = [L1]
    else:
        groups[1]["right_ids"] = ["right:attribute:missing"]
    report = EvalReport.model_validate(payload)

    with pytest.raises(ValueError, match="cover every source exactly once|wrong side"):
        validate_eval_report(report, left, right)


@pytest.mark.parametrize(
    ("status", "left_ids", "right_ids"),
    [
        ("match", [L1], []),
        ("uncertain", [], [R1]),
        ("independent", [L1], [R1]),
        ("independent", [], []),
    ],
)
def test_eval_group_rejects_invalid_side_shape(
    status: str,
    left_ids: list[str],
    right_ids: list[str],
) -> None:
    with pytest.raises(ValidationError):
        EvalGroup.model_validate(
            {
                "status": status,
                "left_ids": left_ids,
                "right_ids": right_ids,
                "reason": "invalid",
            }
        )


def test_eval_report_schema_forbids_canonical_items() -> None:
    payload = report_payload()
    payload["groups"][0]["item"] = attribute("invented").model_dump(mode="json")
    with pytest.raises(ValidationError):
        EvalReport.model_validate(payload)


def test_eval_graph_retries_incomplete_report() -> None:
    incomplete = report_payload()
    incomplete["groups"] = incomplete["groups"][:1]
    model = ScriptedModel(
        [
            [call("submit_eval_report", incomplete, "bad-report")],
            [call("submit_eval_report", report_payload(), "good-report")],
        ]
    )

    result = EvalAgent(model).invoke(*collections())  # type: ignore[arg-type]

    assert len(result.groups) == 2
    assert len(model.messages_seen) == 2
    assert any(
        "rejected" in str(message.content)
        for message in model.messages_seen[-1]
    )


def test_empty_inputs_allow_empty_report() -> None:
    empty = CriteriaAttributeSet(source_item_ids=["empty"], criteria=[], attributes=[])
    report = EvalReport(groups=[])
    validate_eval_report(report, empty, empty)
