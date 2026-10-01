import asyncio
import json
from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from shop_agent_langgraph.agents.eval.graph import EvalAgent
from shop_agent_langgraph.agents.eval.io import load_actual, load_gold, save_report
from shop_agent_langgraph.agents.eval.schemas import FieldScore, MatchingReport, SchemaReview
from shop_agent_langgraph.agents.eval.tools import review_fields, source_references, validate_eval_report, validate_schema_review
from shop_agent_langgraph.domain.criteria import CriteriaAttributeSet
from test_relation_agent import ScriptedChatModel, call


def boolean(item_id, name, description):
    return dict(id=item_id, name=name, description=description, aliases=[], type="boolean")


def numeric(item_id="runtime", criterion=False, units=None):
    item = dict(id=item_id, name="续航", description="依靠自带电池持续工作的时间",
                aliases=[], type="numeric", units=units or ["h"], formula=None)
    if criterion:
        item["direction"] = {"type": "larger_better"}
    return item


def collection(*, criteria=(), attributes=(), ids=("p1",)):
    return CriteriaAttributeSet(source_item_ids=list(ids), criteria=list(criteria), attributes=list(attributes))


def pair():
    return (collection(attributes=[boolean("canonical_vibration", "震动", "提供游戏震动反馈"),
                                   boolean("adjustable", "震动可调", "可以调节震动强度")]),
            collection(attributes=[boolean("different_id", "振动反馈", "提供游戏振动反馈"),
                                   boolean("rgb", "RGB灯效", "具备RGB氛围灯")]))


def matching():
    return dict(match=[dict(gold_ids=["gold:0"], actual_ids=["actual:0"], reason="同为震动反馈")],
                missing=[dict(ref="gold:1", reason="没有震动调节维度")],
                extra=[dict(ref="actual:1", reason="gold没有RGB维度")])


def review(ref="gold:0", fields=("kind", "type", "aliases"), failed=()):
    return dict(gold_ref=ref, fields=[dict(field=f, reason="测试语义判断", score=int(f not in failed)) for f in fields])


class RecordingModel(ScriptedChatModel):
    messages_seen: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.messages_seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def model_for(match_payload, review_payload=None):
    responses = [call("MatchingReport", match_payload)]
    if review_payload is not None:
        responses.append(call("SchemaReview", review_payload))
    return RecordingModel(responses=responses)


@pytest.mark.parametrize("async_run", [False, True])
def test_two_stages_cover_both_sides_and_do_not_mutate_inputs(async_run):
    gold, actual = pair()
    before = [gold.model_dump(), actual.model_dump()]
    model = model_for(matching(), {"items": [review(failed=("aliases",))]})
    agent = EvalAgent(model)
    result = asyncio.run(agent.ainvoke(gold, actual)) if async_run else agent.invoke(gold, actual)
    assert before == [gold.model_dump(), actual.model_dump()]
    assert result.metrics.missing_count == result.metrics.extra_count == 1
    assert result.metrics.matched_gold_count == result.metrics.matched_actual_count == 1
    assert result.metrics.field_scores["type"] == 1
    assert "aliases" not in result.metrics.field_scores
    assert result.metrics.classification_error_count == 0
    assert result.metrics.field_scores["units"] is None
    assert len(model.messages_seen) == 2
    first = json.loads(next(m.content for m in model.messages_seen[0] if m.type == "human"))
    assert all(set(item) == {"name", "description"} for item in first["items"].values())
    assert set(first["items"]) == {"gold:0", "gold:1", "actual:0", "actual:1"}
    assert "canonical_vibration" not in json.dumps(first)
    second = json.loads(next(m.content for m in model.messages_seen[1] if m.type == "human"))
    assert set(second["groups"][0]["gold"]) == {"gold:0"}
    assert set(second["groups"][0]["actual"]) == {"actual:0"}
    assert second["groups"][0]["gold"]["gold:0"]["kind"] == "attribute"
    assert next(s.score for s in result.reviews[0].fields if s.field == "granularity") == 1


def test_merged_items_remain_matched_but_each_gold_item_loses_granularity():
    gold, actual = pair()
    actual = collection(attributes=[boolean("composite", "震动功能", "提供震动反馈并能调节震动强度")])
    payload = dict(match=[dict(gold_ids=["gold:0", "gold:1"], actual_ids=["actual:0"],
                              reason="Market把两个gold维度合并")], missing=[], extra=[])
    model = model_for(payload, {"items": [review("gold:0"), review("gold:1")]})
    result = EvalAgent(model).invoke(gold, actual)
    assert result.metrics.match_group_count == 1
    assert result.metrics.matched_gold_count == 2
    assert result.metrics.matched_actual_count == 1
    assert result.metrics.missing_count == result.metrics.extra_count == 0
    assert result.metrics.granularity_error_count == 2
    assert result.metrics.field_scores["granularity"] == 0
    assert result.metrics.field_scores["type"] == 1


def test_split_actual_items_also_lose_granularity():
    gold = collection(attributes=[boolean("combined", "震动功能", "提供震动反馈和强度调节")])
    actual = collection(attributes=[boolean("rumble", "震动", "提供震动反馈"),
                                   boolean("adjust", "强度可调", "可调节震动强度")])
    payload = dict(match=[dict(gold_ids=["gold:0"], actual_ids=["actual:0", "actual:1"],
                              reason="Market拆分了一个gold条目")], missing=[], extra=[])
    result = EvalAgent(model_for(payload, {"items": [review()]})).invoke(gold, actual)
    assert result.metrics.granularity_error_count == 1
    assert result.metrics.matched_actual_count == 2


def test_cross_kind_match_reports_classification_and_missing_direction_separately():
    gold = collection(criteria=[numeric(criterion=True)])
    actual = collection(attributes=[numeric("duration", units=["min"])])
    payload = dict(match=[dict(gold_ids=["gold:0"], actual_ids=["actual:0"], reason="都是续航")], missing=[], extra=[])
    fields = ("kind", "type", "units", "formula", "direction", "aliases")
    result = EvalAgent(model_for(payload, {"items": [review(fields=fields, failed=("kind", "direction"))]})).invoke(gold, actual)
    scores = {s.field: s.score for s in result.reviews[0].fields}
    assert scores["units"] == 1  # semantic judge can accept convertible units
    assert scores["kind"] == scores["direction"] == 0
    assert scores["granularity"] == 1
    assert result.metrics.classification_error_count == 1
    assert result.metrics.by_kind["criterion"].matched_gold_count == 1
    assert result.metrics.by_kind["attribute"].matched_actual_count == 1
    assert result.metrics.missing_count == result.metrics.extra_count == 0


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_side", "unknown"])
def test_matching_requires_exact_coverage_and_correct_sides(problem):
    gold, actual = pair()
    payload = matching()
    if problem == "missing":
        payload["missing"] = []
    elif problem == "duplicate":
        payload["match"][0]["gold_ids"].append("gold:1")
    elif problem == "wrong_side":
        payload["match"][0]["gold_ids"] = ["actual:0"]
        payload["match"][0]["actual_ids"] = ["gold:0"]
    else:
        payload["extra"][0]["ref"] = "actual:99"
    with pytest.raises(ValueError, match="every source exactly once|wrong side"):
        validate_eval_report(MatchingReport.model_validate(payload), gold, actual)


@pytest.mark.parametrize("score", [True, False, 0.5, -1, 2, "1"])
def test_scores_must_be_binary_integers(score):
    with pytest.raises(ValidationError):
        FieldScore(field="type", score=score, reason="invalid")


def test_invalid_matching_gets_one_bounded_correction():
    gold, actual = pair()
    incomplete = matching()
    incomplete["extra"] = []
    model = RecordingModel(responses=[call("MatchingReport", incomplete),
        call("MatchingReport", matching()), call("SchemaReview", {"items": [review()]})])
    result = EvalAgent(model).invoke(gold, actual)
    assert result.metrics.extra_count == 1
    assert model.position == 3
    assert any("rejected" in str(m.content) for m in model.messages_seen[1])


@pytest.mark.parametrize("failure", ["incomplete", "duplicate_field", "kind_contradiction", "unknown_gold"])
def test_invalid_review_gets_one_correction_and_then_fails(failure):
    gold, actual = pair()
    item = review()
    if failure == "incomplete":
        item["fields"].pop()
    elif failure == "duplicate_field":
        item["fields"].append(item["fields"][0])
    elif failure == "kind_contradiction":
        item["fields"][0]["score"] = 0
    else:
        item["gold_ref"] = "gold:1"  # missing gold items must not get schema review
    invalid = {"items": [item]}
    model = RecordingModel(responses=[call("MatchingReport", matching()),
        call("SchemaReview", invalid), call("SchemaReview", invalid)])
    with pytest.raises(ValueError):
        EvalAgent(model).invoke(gold, actual)
    assert model.position == 3


def test_scope_mismatch_rejected_before_model_call():
    gold, actual = pair()
    actual.source_item_ids = ["other"]
    model = RecordingModel(responses=[])
    with pytest.raises(ValueError, match="same fixed"):
        EvalAgent(model).invoke(gold, actual)
    assert model.position == 0


def test_empty_collections_have_no_fabricated_perfect_score():
    empty = collection()
    model = RecordingModel(responses=[])
    report = EvalAgent(model).invoke(empty, empty)
    assert report.match == report.missing == report.extra == report.reviews == []
    assert all(value is None for value in report.metrics.field_scores.values())
    assert model.position == 0


def test_no_matches_skips_schema_judge():
    gold = collection(attributes=[boolean("a", "震动", "震动反馈")])
    actual = collection()
    payload = dict(match=[], missing=[dict(ref="gold:0", reason="Market无任何维度")], extra=[])
    model = model_for(payload)
    report = EvalAgent(model).invoke(gold, actual)
    assert model.position == 1
    assert report.metrics.missing_count == 1
    assert report.reviews == []


@pytest.mark.parametrize("problem", ["item_id", "product_id"])
def test_duplicate_input_ids_rejected(problem):
    gold, actual = pair()
    if problem == "item_id":
        gold.attributes[1].id = gold.attributes[0].id
    else:
        gold.source_item_ids = ["p1", "p1"]
    with pytest.raises(ValueError, match="duplicate"):
        source_references(gold, actual)


def test_gold_loader_strips_annotations_preserves_review_status_and_exports_report(tmp_path):
    root = Path(__file__).resolve().parents[1]
    gold, metadata = load_gold(root / "eval/cases/gamepad/gold.json")
    assert len(gold.source_item_ids) == 9
    assert len(gold.criteria) == 7 and len(gold.attributes) == 55
    assert metadata["status"] == "draft_pending_human_review"
    assert len(metadata["gold_sha256"]) == 64
    actual_path = tmp_path / "market.json"
    actual_path.write_text(json.dumps({"result": {
        "status": "completed", "item_ids": gold.source_item_ids,
        "criteria": [i.model_dump() for i in gold.criteria],
        "attributes": [i.model_dump() for i in gold.attributes],
    }}), encoding="utf-8")
    actual = load_actual(actual_path)
    assert actual == gold
    empty = collection()
    report = EvalAgent(RecordingModel(responses=[])).invoke(empty, empty)
    report.gold_metadata = metadata
    json_path, md_path = save_report(report, tmp_path / "reports")
    assert json.loads(json_path.read_text(encoding="utf-8"))["gold_metadata"]["status"] == metadata["status"]
    assert "match" in md_path.read_text(encoding="utf-8")


def test_pending_actual_is_not_treated_as_empty_success(tmp_path):
    path = tmp_path / "pending.json"
    path.write_text(json.dumps({"status": "pending", "item_ids": ["p1"], "criteria": [], "attributes": []}))
    with pytest.raises(ValueError, match="completed"):
        load_actual(path)


def test_missing_direction_cannot_be_accepted_as_correct():
    gold = collection(criteria=[numeric(criterion=True)])
    actual = collection(attributes=[numeric()])
    items, _, _ = source_references(gold, actual)
    matching = MatchingReport(match=[dict(gold_ids=["gold:0"], actual_ids=["actual:0"], reason="续航")], missing=[], extra=[])
    fields = review_fields("gold:0", matching.match[0], items)
    invalid = SchemaReview(items=[review(fields=fields, failed=("kind",))])
    with pytest.raises(ValueError, match="direction.*absent"):
        validate_schema_review(invalid, matching, items)


def test_repeated_sync_calls_use_sync_model_transport():
    class SyncOnlyModel(RecordingModel):
        async def _agenerate(self, *args, **kwargs):
            raise AssertionError("sync evaluation must not reuse an async client across event loops")

    gold, actual = pair()
    responses = [call("MatchingReport", matching()), call("SchemaReview", {"items": [review()]})]
    agent = EvalAgent(SyncOnlyModel(responses=responses * 2))
    assert agent.invoke(gold, actual).metrics.missing_count == 1
    assert agent.invoke(gold, actual).metrics.missing_count == 1


def test_mixed_group_does_not_require_other_dimensions_fields():
    gold = collection(criteria=[numeric(criterion=True)], attributes=[boolean("rumble", "震动", "震动反馈")])
    items, _, _ = source_references(gold, gold)
    group = MatchingReport(match=[dict(gold_ids=["gold:0", "gold:1"], actual_ids=["actual:0", "actual:1"], reason="复合")], missing=[], extra=[]).match[0]
    assert set(review_fields("gold:1", group, items)) == {"kind", "type", "aliases"}
    assert "direction" in review_fields("gold:0", group, items)
