from copy import deepcopy

import pytest

from shop_agent_langgraph.agents.eval.graph import EvalAgent
from shop_agent_langgraph.evaluation import rubric_checks as rc
from shop_agent_langgraph.evaluation.rubric import GoldOverlay, load_rubrics
from test_eval_agent import collection, matching, model_for, numeric, pair, review

CTX = rc.CheckContext()


def statuses(units):
    return {u.status for u in units}


def check(rubric_id, sample, ctx=CTX):
    return rc.CHECKERS[rubric_id](sample, ctx)


def assert_passes(rubric_id, sample, ctx=CTX):
    units = check(rubric_id, sample, ctx)
    assert "pass" in statuses(units) and statuses(units) <= {"pass", "na"}, units


def assert_fails(rubric_id, sample, ctx=CTX):
    assert "fail" in statuses(check(rubric_id, sample, ctx)), check(rubric_id, sample, ctx)


# --- text normalization ------------------------------------------------------------

def test_source_comparison_removes_every_whitespace_character():
    assert rc.squash("1000 Hz　高\n刷 回报率​") == "1000Hz高刷回报率"
    assert rc.in_source("1000Hz 高刷\n回报率", "● 1000 Hz　高刷 回报 率")
    assert not rc.in_source("", "anything") and not rc.in_source("2000Hz", "1000Hz")


# --- Research -------------------------------------------------------------------------

CONTEXT = "● 1000 Hz 高刷回报率\n续航 20 小时\n无线 2.4G"


def research(evidence=None, relevance="relevant", item_id="p1"):
    evidence = evidence if evidence is not None else [
        {"subject": "product", "name": "回报率", "value": "1000", "unit": "Hz", "source_text": "1000Hz高刷回报率"}]
    return {"item_id": "p1", "context": CONTEXT, "reference": {"relevance": "relevant"},
            "result": {"item_id": item_id, "relevance": relevance, "relevance_reason": "x", "evidence": evidence}}


def test_rs_1_evidence_quotes_are_substrings_of_the_product_text():
    assert_passes("RS-1", research())
    assert_fails("RS-1", research([{"name": "x", "source_text": "回报率 8000Hz"}]))
    assert_fails("RS-1", research([{"name": "x", "source_text": " "}]))


def test_rs_2_fixed_product_is_relevant():
    assert_passes("RS-2", research())
    assert_fails("RS-2", research(relevance="uncertain"))
    assert_fails("RS-2", research(item_id="other"))


def test_rs_3_value_and_unit_appear_in_their_quote():
    assert_passes("RS-3", research())
    assert_fails("RS-3", research([{"name": "n", "value": "8000", "unit": "Hz", "source_text": "1000Hz高刷回报率"}]))
    assert_fails("RS-3", research([{"name": "n", "value": "1000", "unit": "kHz", "source_text": "1000Hz高刷回报率"}]))
    assert statuses(check("RS-3", research([{"name": "n", "source_text": "无线"}]))) == {"na"}


# --- Aggregation ----------------------------------------------------------------------

def overlay(*core, keywords=("价格",), status="draft_pending_human_review"):
    return GoldOverlay(status=status, core_ids=list(core), excluded_keywords=list(keywords))


def agg_sample(gold=None, actual=None, match=None, reviews=None, core=("canonical_vibration",), **extra):
    g, a = pair()
    gold, actual = gold or g, actual or a
    report = EvalAgent(model_for(match or matching(), {"items": reviews or [review()]})).invoke(gold, actual)
    return {"report": report, "actual": actual.model_dump(mode="json"), "input_item_ids": ["p1"], **extra}, \
        rc.CheckContext(overlay=overlay(*core))


def test_ag_1_core_dimensions_are_all_matched_one_to_one():
    sample, ctx = agg_sample()
    assert_passes("AG-1", sample, ctx)
    sample, ctx = agg_sample(core=("canonical_vibration", "adjustable"))
    assert [u.reason for u in check("AG-1", sample, ctx) if u.status == "fail"] == ["core dimension is missing from Market"]
    sample, ctx = agg_sample(core=("not_in_gold",))
    assert statuses(check("AG-1", sample, ctx)) == {"unresolved"}
    assert statuses(check("AG-1", sample, rc.CheckContext())) == {"na"}
    gold = collection(attributes=[numeric("a"), numeric("b")])
    actual = collection(attributes=[numeric("merged")])
    payload = dict(match=[dict(gold_ids=["gold:0", "gold:1"], actual_ids=["actual:0"], reason="合并")], missing=[], extra=[])
    sample, ctx = agg_sample(gold, actual, payload, [review("gold:0", ("kind", "type", "units", "formula", "aliases")),
                                                    review("gold:1", ("kind", "type", "units", "formula", "aliases"))], core=("a",))
    assert [u.reason for u in check("AG-1", sample, ctx)] == ["2 gold to 1 actual"]


def test_ag_3_kind_type_and_field_presence_of_matched_dimensions():
    sample, ctx = agg_sample()
    assert_passes("AG-3", sample, ctx)
    gold = collection(criteria=[numeric(criterion=True)])
    actual = collection(attributes=[numeric("duration", units=["min"])])
    payload = dict(match=[dict(gold_ids=["gold:0"], actual_ids=["actual:0"], reason="都是续航")], missing=[], extra=[])
    fields = ("kind", "type", "units", "formula", "direction", "aliases")
    sample, ctx = agg_sample(gold, actual, payload, [review(fields=fields, failed=("kind", "direction"))], core=("runtime",))
    failed = {u.unit for u in check("AG-3", sample, ctx) if u.status == "fail"}
    assert failed == {"runtime.kind", "runtime.direction"}


def test_ag_3_mixed_groups_are_unresolved_not_guessed():
    gold = collection(attributes=[numeric("a"), numeric("b")])
    actual = collection(criteria=[numeric("x", criterion=True)], attributes=[numeric("y")])
    payload = dict(match=[dict(gold_ids=["gold:0"], actual_ids=["actual:0", "actual:1"], reason="拆分")], missing=[dict(ref="gold:1", reason="无")], extra=[])
    sample, ctx = agg_sample(gold, actual, payload, [review(fields=("kind", "type", "units", "formula", "aliases"))])
    assert "unresolved" in statuses(check("AG-3", sample, ctx))


def test_ag_6_no_excluded_category_in_output_dimensions():
    sample, ctx = agg_sample()
    assert_passes("AG-6", sample, ctx)
    sample["actual"]["attributes"][1]["name"] = "销 量"
    ctx.overlay = overlay("canonical_vibration", keywords=("价格", "销量"))
    assert_fails("AG-6", sample, ctx)
    sample["actual"]["attributes"][1].update(name="外观", aliases=["商品价格"])
    assert_fails("AG-6", sample, ctx)


def test_ag_7_unique_ids_and_exact_source_item_ids():
    sample, ctx = agg_sample()
    assert_passes("AG-7", sample, ctx)
    clone = deepcopy(sample)
    clone["actual"]["attributes"][1]["id"] = clone["actual"]["attributes"][0]["id"]
    assert_fails("AG-7", clone, ctx)
    sample["input_item_ids"] = ["p1", "p2"]
    assert_fails("AG-7", sample, ctx)


# --- orchestration --------------------------------------------------------------------

def test_every_rubric_has_a_checker():
    assert [r.id for r in load_rubrics().rubrics] == list(rc.CHECKERS)


def test_run_module_pools_samples():
    out = rc.run_module("research", {"p1": research(), "p2": research(relevance="uncertain")})
    assert set(out) == {"RS-1", "RS-2", "RS-3"}
    assert {u.unit for u in out["RS-2"]} == {"p1/relevance", "p2/relevance"}
    assert set(rc.run_module("research", {"p1": research()}, only={"RS-1"})) == {"RS-1"}
