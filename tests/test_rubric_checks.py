import json
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from shop_agent_langgraph.agents.eval.graph import EvalAgent
from shop_agent_langgraph.domain.taxonomy import get_children, get_nodes
from shop_agent_langgraph.evaluation import rubric_checks as rc
from shop_agent_langgraph.evaluation.rubric import GoldOverlay, load_rubrics
from test_eval_agent import collection, matching, model_for, numeric, pair, review

CTX = rc.CheckContext()


def node(node_id):
    found, _ = get_nodes([node_id])
    return {k: found[0][k] for k in ("node_id", "node_name", "node_path")}


def route_out(status="resolved", resolved=("301",), candidates=(), children=None):
    if children is None:
        children = [c["node_id"] for c in get_children(list(resolved))[0]["children"]] if resolved else []
    return {"product": "游戏手柄", "status": status, "resolved_nodes": [node(i) for i in resolved],
            "candidates": [node(i) for i in candidates], "children": [node(i) for i in children]}


def call(name, args=None, output=None, error=False):
    return {"name": name, "args": args or {}, "output": json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else output, "error": error}


def e2e(calls, reply="好的，我先帮你梳理游戏手柄的选购维度。", reference=None):
    return {"input": "我想买游戏手柄", "reference": reference or {"action": "create", "node_id": "301"},
            "trace": {"tool_calls": calls, "final_reply": reply}}


def good_chain():
    return [call("call_intent_agent", {"request": "我想买游戏手柄"}, {"action": "create"}),
            call("call_route_agent", {"product": "游戏手柄"}, route_out()),
            call("call_market_agent", {"node_id": "301"}, {"status": "completed"})]


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


def test_internal_tool_names_match_the_supervisor_tools():
    from shop_agent_langgraph.supervisor.tools import SUPERVISOR_TOOLS
    assert {t.name for t in SUPERVISOR_TOOLS} <= set(rc.INTERNAL_TOOL_NAMES)


# --- E2E ------------------------------------------------------------------------------

def test_e2e_1_call_chain():
    assert_passes("E2E-1", e2e(good_chain()))
    ambiguous = [call("call_intent_agent", output={"action": "create"}),
                 call("call_route_agent", output=route_out("ambiguous", (), ("303", "4512")))]
    assert_passes("E2E-1", e2e(ambiguous, reference={"action": "create"}))
    assert_passes("E2E-1", e2e([good_chain()[0]], reference={"action": "query"}))
    assert_passes("E2E-1", e2e(good_chain() + [call("call_relation_agent", {"node_id": "301"}, {})]))
    assert_fails("E2E-1", e2e(good_chain()[:2]))                                  # resolved but market skipped
    assert_fails("E2E-1", e2e(ambiguous + [call("call_market_agent", {"node_id": "303"}, {"status": "completed"})],
                              reference={"action": "create"}))
    assert_fails("E2E-1", e2e(good_chain(), reference={"action": "query"}))
    assert_fails("E2E-1", e2e([good_chain()[1], good_chain()[0], good_chain()[2]]))  # wrong order
    assert statuses(check("E2E-1", e2e(good_chain(), reference={"action": "update"}))) == {"na"}


def test_e2e_2_market_uses_the_routed_node():
    assert_passes("E2E-2", e2e(good_chain()))
    wrong = good_chain()
    wrong[2] = call("call_market_agent", {"node_id": "303"}, {"status": "completed"})
    assert_fails("E2E-2", e2e(wrong))
    assert_fails("E2E-2", e2e([call("call_market_agent", {"node_id": "301"}, {"status": "completed"})]))
    assert statuses(check("E2E-2", e2e(good_chain()[:2]))) == {"na"}


def test_e2e_3_resolved_node_matches_reference_or_allowed_set():
    assert_passes("E2E-3", e2e(good_chain()))
    assert_passes("E2E-3", e2e(good_chain(), reference={"action": "create", "allowed_node_ids": ["301", "543590"]}))
    assert_fails("E2E-3", e2e(good_chain(), reference={"action": "create", "node_id": "303"}))
    assert_fails("E2E-3", e2e(good_chain()[:1] + [call("call_route_agent", output=route_out("ambiguous", (), ("303",)))]))
    assert statuses(check("E2E-3", e2e(good_chain(), reference={"action": "create"}))) == {"na"}


def test_e2e_4_market_completed():
    assert_passes("E2E-4", e2e(good_chain()))
    pending = good_chain()
    pending[2] = call("call_market_agent", {"node_id": "301"}, {"status": "pending"})
    assert_fails("E2E-4", e2e(pending))
    pending[2] = call("call_market_agent", {"node_id": "301"}, "Error: boom", error=True)
    assert_fails("E2E-4", e2e(pending))


@pytest.mark.parametrize("reply", ['{"status": "resolved"}', "我调用了 call_route_agent", "Market Agent 已完成",
                                   "任务 item_id 123", "编号 844285076150", "节点 301 已确定"])
def test_e2e_6_flags_leaked_internals(reply):
    assert_fails("E2E-6", e2e(good_chain(), reply=reply))


def test_e2e_6_allows_natural_replies_and_plain_numbers():
    assert_passes("E2E-6", e2e(good_chain(), reply="预算 301 元以内的游戏手柄，我先整理选购维度。"))
    assert statuses(check("E2E-6", e2e(good_chain(), reply=" "))) == {"unresolved"}


def test_e2e_7_ambiguous_reply_names_every_candidate():
    chain = [call("call_intent_agent", output={"action": "create"}),
             call("call_route_agent", output=route_out("ambiguous", (), ("303", "4512")))]
    assert_passes("E2E-7", e2e(chain, reply="你是想看「键盘」还是 数字 键盘？", reference={"action": "create"}))
    assert_fails("E2E-7", e2e(chain, reply="你是想看键盘吗？", reference={"action": "create"}))
    assert statuses(check("E2E-7", e2e(good_chain()))) == {"na"}


def test_trace_from_messages_extracts_calls_outputs_and_reply():
    messages = [HumanMessage("我想买游戏手柄"),
                AIMessage("", tool_calls=[{"name": "call_route_agent", "args": {"product": "游戏手柄"}, "id": "t1"}]),
                ToolMessage(json.dumps(route_out()), tool_call_id="t1", name="call_route_agent"),
                AIMessage("已确定类别。")]
    trace = rc.trace_from_messages(messages)
    assert [c["name"] for c in trace["tool_calls"]] == ["call_route_agent"]
    assert json.loads(trace["tool_calls"][0]["output"])["status"] == "resolved" and trace["final_reply"] == "已确定类别。"


# --- Intent ---------------------------------------------------------------------------

def intent(text="预算 500，最好无线的", **output):
    base = {"action": "create", "category": "游戏手柄", "criteria_preferences": ["价格 ≤ 500 元"],
            "attribute_preferences": ["连接方式 = 无线"]}
    return {"input": text, "output": {**base, **output}, "reference": {"action": "create", "category": "游戏手柄"}}


def test_int_1_action():
    assert_passes("INT-1", intent())
    assert_fails("INT-1", intent(action="update"))


def test_int_2_category_normalization_and_null():
    assert_passes("INT-2", intent(category=" 游戏 手柄 "))
    assert_fails("INT-2", intent(category="键盘"))
    null_reference = {**intent(category=None), "reference": {"action": "query", "category": None}}
    assert_passes("INT-2", null_reference)
    assert_fails("INT-2", {**null_reference, "output": {**null_reference["output"], "category": "游戏手柄"}})


def test_int_5_numbers_survive_verbatim():
    assert_passes("INT-5", intent())
    assert_passes("INT-5", intent("预算 1,000 元，买 2 个", criteria_preferences=["价格 ≤ 1000 元"], attribute_preferences=["数量 = 2"]))
    assert_fails("INT-5", intent(criteria_preferences=["价格不超过五百元"]))
    assert_fails("INT-5", intent("预算 1500", criteria_preferences=["价格 ≤ 500 元"]))   # 500 is not 1500
    assert statuses(check("INT-5", {**intent(), "reference": {"action": "query"}})) == {"na"}


# --- Route ----------------------------------------------------------------------------

def route_sample(output, **reference):
    return {"input": "游戏手柄", "output": output, "reference": reference}


def test_rt_1_and_rt_2_status_and_resolved_node():
    resolved = route_out()
    assert_passes("RT-1", route_sample(resolved, status="resolved", node_id="301"))
    assert_fails("RT-1", route_sample(resolved, status="ambiguous"))
    assert_passes("RT-2", route_sample(resolved, status="resolved", node_id="301"))
    assert_passes("RT-2", route_sample(resolved, status="resolved", acceptable_node_ids=["301", "543590"]))
    assert_fails("RT-2", route_sample(resolved, status="resolved", node_id="303"))
    assert_fails("RT-2", route_sample(route_out("ambiguous", (), ("303",)), status="resolved", node_id="303"))
    assert statuses(check("RT-2", route_sample(resolved, status="ambiguous"))) == {"na"}


def test_rt_3_nodes_exist_with_matching_name_and_path():
    assert_passes("RT-3", route_sample(route_out()))
    ghost = route_out()
    ghost["candidates"] = [{"node_id": "999999999", "node_name": "不存在", "node_path": "不存在"}]
    assert_fails("RT-3", route_sample(ghost))
    renamed = route_out()
    renamed["resolved_nodes"][0]["node_name"] = "游戏手柄X"
    assert_fails("RT-3", route_sample(renamed))
    wrong_path = route_out()
    wrong_path["children"][0]["node_path"] = "错误路径"
    assert_fails("RT-3", route_sample(wrong_path))


def test_rt_4_reference_node_is_among_ambiguous_candidates():
    ambiguous = route_out("ambiguous", (), ("303", "4512"))
    assert_passes("RT-4", route_sample(ambiguous, status="ambiguous", acceptable_node_ids=["303"]))
    assert_fails("RT-4", route_sample(ambiguous, status="ambiguous", acceptable_node_ids=["74"]))
    assert statuses(check("RT-4", route_sample(route_out(), status="resolved", node_id="301"))) == {"na"}


def test_rt_5_children_are_exact_direct_children():
    assert_passes("RT-5", route_sample(route_out()))
    missing = route_out()
    missing["children"] = missing["children"][1:]
    assert_fails("RT-5", route_sample(missing))
    extra = route_out()
    extra["children"].append(node("303"))
    assert_fails("RT-5", route_sample(extra))
    assert statuses(check("RT-5", route_sample(route_out("ambiguous", (), ("303",))))) == {"na"}


# --- Market screening -----------------------------------------------------------------

PAGE = "## 参数\n续航：20 小时\n回报率 1000 Hz"


def screening(item_id, *, selected=True, pages=("page-0001",), quote="续航：20小时", page_id="page-0001", full=False):
    return {"stage": "screening", "item_id": item_id, "relevance": "relevant", "reason": "手柄",
            "evidence": [{"page_id": page_id, "source_text": quote}], "selected": selected,
            "read_page_ids": list(pages), "page_read_calls": 1, "full_context_read": full}


def market(ids=("a", "b", "c"), audit=None, **reference):
    audit = audit if audit is not None else [screening(i) for i in ids]
    return {"result": {"query": "游戏手柄", "item_ids": list(ids), "status": "completed", "audit": audit},
            "reference": reference, "page_texts": {i: {"page-0001": PAGE} for i in "abcdef"}}


def test_ms_1_selected_products_were_searched_read_and_unique():
    assert_passes("MS-1", market())
    unread = market(audit=[screening("a", pages=()), screening("b"), screening("c")])
    assert_fails("MS-1", unread)
    assert_passes("MS-1", market(audit=[screening("a", pages=(), full=True), screening("b"), screening("c")]))
    unsearched = market()
    unsearched["result"]["item_ids"].append("zzz")
    assert_fails("MS-1", unsearched)
    dup = market(audit=[screening("a"), screening("a"), screening("b")], ids=("a", "b"))
    assert_fails("MS-1", dup)
    delivered_unselected = market(audit=[screening("a"), screening("b"), screening("c", selected=False)])
    assert_fails("MS-1", delivered_unselected)


def test_ms_2_screening_quotes_come_from_read_pages():
    assert_passes("MS-2", market())
    assert_fails("MS-2", market(audit=[screening("a", quote="续航：40小时"), screening("b"), screening("c")]))
    assert_fails("MS-2", market(audit=[screening("a", page_id="page-0002"), screening("b"), screening("c")]))
    other_page = [screening("a", pages=("page-0001", "page-0002"), page_id="page-0002"), screening("b"), screening("c")]
    assert "unresolved" in statuses(check("MS-2", market(audit=other_page)))   # page text unavailable


def test_ms_3_no_reference_irrelevant_product_selected():
    assert_passes("MS-3", market(irrelevant_item_ids=["x"]))
    assert_fails("MS-3", market(irrelevant_item_ids=["b"]))
    assert statuses(check("MS-3", market())) == {"na"}


def test_ms_4_selected_count_within_bounds():
    assert_passes("MS-4", market())
    assert_passes("MS-4", market(ids=("a", "b", "c", "d")))
    assert_fails("MS-4", market(ids=("a", "b")))
    assert_fails("MS-4", market(ids=("a", "b", "c", "d", "e")))


def test_ms_5_recall_of_reference_relevant_products():
    assert_passes("MS-5", market(relevant_item_ids=["a", "b"]))
    assert_fails("MS-5", market(relevant_item_ids=["a", "q"]))
    assert statuses(check("MS-5", market())) == {"na"}


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


def test_ag_1_core_dimensions_are_all_matched():
    sample, ctx = agg_sample()
    assert_passes("AG-1", sample, ctx)
    sample, ctx = agg_sample(core=("canonical_vibration", "adjustable"))
    assert_fails("AG-1", sample, ctx)
    sample, ctx = agg_sample(core=("not_in_gold",))
    assert statuses(check("AG-1", sample, ctx)) == {"unresolved"}
    assert statuses(check("AG-1", sample, rc.CheckContext())) == {"na"}


def test_ag_2_non_core_coverage():
    sample, ctx = agg_sample(core=("adjustable",))           # non-core vibration is matched
    assert_passes("AG-2", sample, ctx)
    sample, ctx = agg_sample(core=("canonical_vibration",))  # non-core "adjustable" is missing
    assert_fails("AG-2", sample, ctx)


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


def test_ag_4_core_dimensions_match_one_to_one():
    sample, ctx = agg_sample()
    assert_passes("AG-4", sample, ctx)
    gold = collection(attributes=[numeric("a"), numeric("b")])
    actual = collection(attributes=[numeric("merged")])
    payload = dict(match=[dict(gold_ids=["gold:0", "gold:1"], actual_ids=["actual:0"], reason="合并")], missing=[], extra=[])
    sample, ctx = agg_sample(gold, actual, payload, [review("gold:0", ("kind", "type", "units", "formula", "aliases")),
                                                    review("gold:1", ("kind", "type", "units", "formula", "aliases"))], core=("a",))
    assert_fails("AG-4", sample, ctx)
    sample, ctx = agg_sample(core=("adjustable",))           # missing core dimension is not one-to-one either
    assert_fails("AG-4", sample, ctx)


def test_ag_5_reads_existing_report_field_scores():
    gold = collection(criteria=[numeric(criterion=True)])
    actual = collection(criteria=[numeric("runtime", criterion=True)])
    payload = dict(match=[dict(gold_ids=["gold:0"], actual_ids=["actual:0"], reason="续航")], missing=[], extra=[])
    fields = ("kind", "type", "units", "formula", "direction", "aliases")
    good, ctx = agg_sample(gold, actual, payload, [review(fields=fields)], core=("runtime",))
    assert_passes_ag5 = rc.ag_5(good, ctx)
    assert statuses(assert_passes_ag5) == {"pass"} and {u.unit for u in assert_passes_ag5} == {"runtime.units", "runtime.formula", "runtime.direction"}
    bad, ctx = agg_sample(gold, actual, payload, [review(fields=fields, failed=("units",))], core=("runtime",))
    assert [u.unit for u in rc.ag_5(bad, ctx) if u.status == "fail"] == ["runtime.units"]


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

def test_every_code_rubric_has_a_checker_and_llm_rubrics_need_a_judge():
    rubrics = load_rubrics()
    assert {r.id for r in rubrics.rubrics if r.judge == "code"} == set(rc.CHECKERS)
    assert set(rc.BUILTIN_JUDGES) == {"AG-5"}


def test_run_module_pools_samples_and_skips_llm_rubrics_without_a_judge():
    out = rc.run_module("research", {"p1": research(), "p2": research(relevance="uncertain")})
    assert set(out) == {"RS-1", "RS-2", "RS-3"}
    assert {u.unit for u in out["RS-2"]} == {"p1/relevance", "p2/relevance"}
    judged = rc.run_module("research", {"p1": research()}, judges={"RS-4": lambda s, c: [rc.UnitResult(unit="e0", status="pass")]})
    assert [u.unit for u in judged["RS-4"]] == ["p1/e0"] and "RS-5" not in judged
