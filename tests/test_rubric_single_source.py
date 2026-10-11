import asyncio
import json

import httpx2
import pytest
import yaml
from typesafe_sdk import AsyncTypeSafeClient

from shop_agent_langgraph.agents.eval.graph import EvalAgent
from shop_agent_langgraph.agents.eval.jev import JevJudge
from shop_agent_langgraph.agents.eval.schemas import SchemaReview
from shop_agent_langgraph.domain.criteria import CriteriaAttributeSet
from shop_agent_langgraph.evaluation import annotations
from shop_agent_langgraph.evaluation import rubric as rb
from test_eval_agent import model_for, review
from test_fixed_market_evaluation import jev_payload, small_case

ALL_FIELDS = ("kind", "type", "units", "formula", "direction", "values", "value_domain", "aliases")


def llm_prompt():
    """The schema-review system prompt the EvalAgent really sends to its model."""
    case = small_case()
    gold = CriteriaAttributeSet.model_validate(case["expected_output"])
    payload = dict(match=[dict(gold_ids=["gold:0"], actual_ids=["actual:0"], reason="续航")], missing=[], extra=[])
    fields = ("kind", "type", "units", "formula", "direction", "aliases")
    model = model_for(payload, {"items": [review(fields=fields)]})
    report = EvalAgent(model).invoke(gold, gold)
    system = next(m for m in model.messages_seen[1] if m.type == "system")
    return system.content, report, case


def jev_instructions():
    seen = {}

    def transport(request):
        body = json.loads(request.content)
        answers = {}
        for field, question in body["questions"].items():
            seen[field] = question["instructions"]
            answers[field] = {"type": "choice", "choice": "correct", "confidence": .9,
                              "probabilities": {"correct": .9, "incorrect": .05, "unclear": .05}}
        return httpx2.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 1}, "answers": answers})

    judge = JevJudge(client_factory=lambda: AsyncTypeSafeClient(api_key="k", transport=httpx2.MockTransport(transport)))
    baseline = SchemaReview(items=[review(fields=("kind", "type", "units", "formula", "direction", "aliases"))])
    asyncio.run(judge.review(jev_payload(), baseline))
    return seen


def human_questions():
    _, report, case = llm_prompt()
    return {t["field"]: t["input"]["question"] for t in annotations.review_tasks(case, report) if t["task_kind"] == "schema_field"}


def test_yaml_defines_every_schema_field_once():
    definitions = rb.field_definitions()
    assert tuple(definitions) == ALL_FIELDS and all(definitions.values())
    assert set(rb.review_questions()) == set(ALL_FIELDS)
    assert annotations.REVIEW_VERSION == "v2"


def test_prompt_jev_and_human_wording_all_render_from_the_yaml():
    definitions = rb.field_definitions()
    prompt, _, _ = llm_prompt()
    assert "{{" not in prompt and all(f"- {name}: {text}" in prompt for name, text in definitions.items())
    jev = jev_instructions()
    assert set(jev) == {"units", "direction", "aliases"} and all(definitions[f] in jev[f] for f in jev)
    human = human_questions()
    assert set(human) >= {"kind", "type", "units", "formula", "direction", "aliases"}
    assert all(definitions[f] in human[f] and f"`{f}`" in human[f] for f in human)


def test_editing_a_yaml_definition_changes_prompt_jev_and_human_question_together(tmp_path, monkeypatch):
    before = (llm_prompt()[0], jev_instructions()["direction"], human_questions()["direction"])
    task_ids_before = {t["id"] for t in annotations.review_tasks(small_case(), llm_prompt()[1]) if t["field"] == "direction"}
    path = tmp_path / "schema_fields.yaml"
    data = yaml.safe_load(rb.SCHEMA_FIELDS_PATH.read_text(encoding="utf-8"))
    new = "ZZ-NEW direction wording: compare preference semantics only."
    next(f for f in data["fields"] if f["name"] == "direction")["definition"] = new
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(rb, "SCHEMA_FIELDS_PATH", path)
    after = (llm_prompt()[0], jev_instructions()["direction"], human_questions()["direction"])
    for old, changed in zip(before, after):
        assert new not in old and new in changed
    # every other field's wording is untouched
    other = "- units: " + rb.field_definitions()["units"]
    assert other in after[0]
    assert {t["id"] for t in annotations.review_tasks(small_case(), llm_prompt()[1]) if t["field"] == "direction"} != task_ids_before


def test_prompt_template_needs_exactly_one_placeholder():
    with pytest.raises(ValueError, match="exactly once"):
        rb.render_schema_prompt("no placeholder")
    with pytest.raises(ValueError, match="exactly once"):
        rb.render_schema_prompt("{{FIELD_DEFINITIONS}} {{FIELD_DEFINITIONS}}")


def test_a_field_cannot_be_defined_twice(tmp_path):
    path = tmp_path / "schema_fields.yaml"
    data = yaml.safe_load(rb.SCHEMA_FIELDS_PATH.read_text(encoding="utf-8"))
    data["fields"].append({"name": "units", "definition": "duplicate"})
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate field definitions"):
        rb.field_definitions(path)
