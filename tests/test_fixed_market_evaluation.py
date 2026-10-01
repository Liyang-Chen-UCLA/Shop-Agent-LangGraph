import asyncio
from copy import deepcopy
import json
from pathlib import Path

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient

from shop_agent_langgraph.agents.eval.jev import JevJudge
from shop_agent_langgraph.agents.eval.schemas import SchemaReview
from shop_agent_langgraph.agents.market.graph import MarketAgent
from shop_agent_langgraph.agents.research.graph import ResearchAgent
from shop_agent_langgraph.agents.research.schemas import ResearchResult
from shop_agent_langgraph.domain.criteria import CriteriaAttributeSet
from shop_agent_langgraph.evaluation.annotations import review_tasks, validate_reviews, export_candidates
from shop_agent_langgraph.evaluation.cases import digest, load_case, prepare_case, seal_case, validate_case, write_json
from shop_agent_langgraph.evaluation.experiment import report_scores, run_experiment
from test_eval_agent import numeric, review
from test_eval_agent import RecordingModel, call


def small_case():
    item = numeric(criterion=True)
    gold = dict(source_item_ids=["p1"], criteria=[item], attributes=[])
    return seal_case(dict(input=dict(target="独立手柄", source_item_ids=["p1"], product_contexts={"p1": "原始续航证据"}),
        expected_output=gold, metadata={"status": "draft_pending_human_review"},
        gold_annotation={**deepcopy(gold), "annotation_metadata": {"status": "draft_pending_human_review"}}))


def test_real_case_freezes_nine_products_and_rejects_tampering(tmp_path):
    case = prepare_case(Path(__file__).resolve().parents[1] / "eval/cases/gamepad/gold.json")
    assert len(case["input"]["source_item_ids"]) == 9
    assert len(validate_case(case).attributes) == 55
    path = write_json(tmp_path / "case.json", case)
    assert load_case(path) == case
    case["input"]["product_contexts"][case["input"]["source_item_ids"][0]] += "修改"
    with pytest.raises(ValueError, match="hash changed"):
        validate_case(case)


def test_snapshot_research_does_not_read_live_product_store(monkeypatch):
    monkeypatch.setattr("shop_agent_langgraph.agents.research.graph.get_product_raw_text", lambda _: pytest.fail("live lookup"))
    state = ResearchAgent._input("p1", "独立手柄", raw_text="冻结的证据")
    assert "冻结的证据" in state["messages"][0].content


@pytest.mark.parametrize("relevance", ["relevant", "uncertain", "irrelevant"])
def test_fixed_market_never_selects_or_silently_drops_products(relevance):
    class Research:
        async def ainvoke(self, item_id, target, **kwargs):
            assert kwargs["raw_text"] == item_id
            return ResearchResult(item_id=item_id, relevance=relevance, evidence=[])

    class Aggregation:
        async def ainvoke(self, results, target, **kwargs):
            return CriteriaAttributeSet(source_item_ids=[r.item_id for r in results], criteria=[], attributes=[])

    agent = MarketAgent(None, Aggregation())
    ids = [f"p{i}" for i in range(9)]
    call = agent.ainvoke_fixed_products(ids, "独立手柄", product_contexts={i: i for i in ids}, research=Research())
    if relevance == "relevant":
        result = asyncio.run(call)
        assert result.item_ids == ids and result.status == "completed"
    else:
        with pytest.raises(ValueError, match="did not confirm"):
            asyncio.run(call)


def test_fixed_market_rejects_bad_scope_before_research():
    with pytest.raises(ValueError, match="snapshot"):
        asyncio.run(MarketAgent(None, None).ainvoke_fixed_products(["p1"], "独立手柄", product_contexts={}, research=None))


def jev_payload():
    gold = dict(kind="criterion", item=numeric(criterion=True))
    actual = deepcopy(gold)
    actual["item"]["units"] = ["min"]
    return [dict(gold={"gold:0": gold}, actual={"actual:0": actual},
                 required_fields={"gold:0": ["kind", "type", "units", "formula", "direction", "aliases"]})]


@pytest.mark.parametrize("choice", ["correct", "incorrect", "unclear"])
def test_jev_sdk_serialization_binary_choices_and_no_baseline_overwrite(choice):
    seen = []

    def transport(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["model"] == "jev-1.13.0"
        assert set(body["state"]) == {"gold", "actual"}
        answers = {}
        for field in body["questions"]:
            assert body["questions"][field]["type"] == "choice"
            answers[field] = {"type": "choice", "choice": choice, "confidence": .95,
                "probabilities": {key: (.96 if key == choice else .02) for key in ("correct", "incorrect", "unclear")}}
        return httpx2.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 100}, "answers": answers})

    judge = JevJudge(client_factory=lambda: AsyncTypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(transport)))
    baseline = SchemaReview(items=[review(fields=("kind", "type", "units", "formula", "direction", "aliases"))])
    before = baseline.model_dump()
    result = asyncio.run(judge.review(jev_payload(), baseline))
    assert baseline.model_dump() == before
    assert len(seen) == 1
    assert set(seen[0]["questions"]) == {"units", "direction", "aliases"}
    decision = next(d for d in result["decisions"] if d["field"] == "units")
    assert decision["candidate_score"] == {"correct": 1, "incorrect": 0}.get(choice)
    assert decision["status"] == "needs_review"  # uncalibrated, no invented default threshold
    assert result["calibrated"] is False


def test_jev_missing_key_fails_before_any_market_work(monkeypatch, tmp_path):
    monkeypatch.setattr("shop_agent_langgraph.agents.eval.jev.load_environment", lambda: None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        run_experiment(small_case(), tmp_path / "run", jev=JevJudge())
    assert not (tmp_path / "run").exists()


def test_review_tasks_are_stable_and_completion_does_not_auto_approve_gold(tmp_path):
    case = small_case()
    tasks = review_tasks(case)
    assert tasks == review_tasks(case)
    assert {t["task_kind"] for t in tasks} == {"gold_item", "gold_collection"}
    correction = deepcopy(tasks[0]["output"])
    correction["description"] = "人工修正的续航定义"
    reviews = [dict(task=tasks[0], human_score=0, correction=correction, score_id="human-score")]
    original = deepcopy(case)
    validate_reviews(case, reviews)
    export_candidates(case, reviews, tmp_path)
    assert case == original
    candidate = load_case(tmp_path / "case-candidate.json")
    assert candidate["metadata"]["status"] == "draft_pending_human_review"
    assert candidate["expected_output"]["criteria"][0]["description"] == "人工修正的续航定义"
    assert candidate["metadata"]["expected_output_sha256"] != case["metadata"]["expected_output_sha256"]


def test_invalid_evidence_correction_is_rejected():
    case = small_case()
    task = review_tasks(case)[0]
    correction = {**task["output"], "evidence": [{"item_id": "p1", "quote": "不存在的原文"}]}
    with pytest.raises(ValueError, match="quote"):
        validate_reviews(case, [dict(task=task, human_score=0, correction=correction)])


def test_complete_local_delivery_replay_scores_and_persists_baseline(tmp_path):
    case = small_case()
    actual = write_json(tmp_path / "actual.json", case["expected_output"])
    model = RecordingModel(responses=[call("MatchingReport", dict(match=[dict(
        gold_ids=["gold:0"], actual_ids=["actual:0"], reason="续航")], missing=[], extra=[])),
        call("SchemaReview", {"items": [review(fields=("units", "direction", "aliases"))]})])
    manifest = run_experiment(case, tmp_path / "run", actual_path=actual, model=model)
    assert manifest["status"] == "completed"
    assert manifest["metadata"]["run_mode"] == "existing-delivery"
    assert manifest["metrics"]["field_scores"]["kind"] == 1
    assert (tmp_path / "run/report.md").exists()
    report = json.loads((tmp_path / "run/report.json").read_text(encoding="utf-8"))
    assert report["gold_metadata"]["status"] == "draft_pending_human_review"


def test_failed_task_is_not_reported_as_success(tmp_path):
    case = small_case()
    actual = write_json(tmp_path / "actual.json", {**case["expected_output"], "source_item_ids": ["wrong"]})
    with pytest.raises(ValueError, match="fixed product"):
        run_experiment(case, tmp_path / "run", actual_path=actual, model=RecordingModel(responses=[]))
    manifest = json.loads((tmp_path / "run/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and "metrics" not in manifest


def test_review_publishing_retries_do_not_duplicate_queue_items(monkeypatch, tmp_path):
    from contextlib import contextmanager
    import shop_agent_langgraph.evaluation.annotations as module
    remote_items = []
    queue = {"id": "queue", "name": "market-dimension-review", "scoreConfigIds": ["config"]}
    config = {"id": "config", "name": "human_correct", "dataType": "BOOLEAN"}

    def list_items(resource, action, *args):
        return [config] if resource == "score-configs" else [queue] if action == "list" else remote_items

    def api(resource, action, *args, body=None):
        assert action == "create-item"
        remote_items.append(body)
        return body

    class Client:
        calls = 0

        @contextmanager
        def start_as_current_observation(self, **kwargs):
            self.calls += 1
            yield type("Span", (), {"trace_id": "trace", "id": f"observation-{self.calls}"})()

        def flush(self):
            pass

    monkeypatch.setattr(module, "all_items", list_items)
    monkeypatch.setattr(module, "api", api)
    client = Client()
    tasks = review_tasks(small_case())
    path = tmp_path / "ledger.json"
    module.publish_tasks(tasks, client=client, ledger_path=path)
    module.publish_tasks(tasks, client=client, ledger_path=path)
    assert client.calls == len(remote_items) == len(tasks)


def test_dataset_draft_roundtrip_preserves_snapshot_and_annotation():
    from types import SimpleNamespace
    from shop_agent_langgraph.evaluation.datasets import dataset_draft, case_from_dataset
    case = small_case()
    draft = dataset_draft(case)
    item = draft["item"]
    hosted = SimpleNamespace(items=[SimpleNamespace(input=item["input"],
        expected_output=item["expectedOutput"], metadata=item["metadata"])])
    assert case_from_dataset(hosted) == case
    assert dataset_draft(case)["item"]["id"] == item["id"]
    corrupted = deepcopy(hosted)
    corrupted.items[0].input["product_contexts"]["p1"] += " changed"
    with pytest.raises(ValueError, match="hash changed"):
        case_from_dataset(corrupted)


def test_calibration_hides_human_label_and_never_auto_approves(tmp_path):
    from shop_agent_langgraph.evaluation.calibration import calibrate

    class Judge:
        model = "jev-test-pinned"

        def preflight(self):
            pass

        async def review(self, groups, baseline):
            assert set(groups[0]) == {"gold", "actual", "required_fields"}
            assert "human_score" not in json.dumps(groups)
            assert "expected_output" not in json.dumps(groups)
            return {"decisions": [dict(field="units", candidate_score=1)]}

    sample = dict(task_id="human-task", field="units", input=jev_payload()[0], expected_output=0)
    # Calibration state is a single gold item, not a keyed group.
    sample["input"]["gold"] = sample["input"]["gold"]["gold:0"]
    result = calibrate([sample], Judge(), tmp_path / "calibration")
    assert result["status"] == "completed" and result["approved"] is False
    assert result["by_field"]["units"] == dict(count=1, correct=0, unclear=0,
                                               positive_labels=0, negative_labels=1)


def test_comparison_rejects_changed_gold_and_omits_na_deltas(tmp_path):
    from shop_agent_langgraph.evaluation.calibration import compare_runs
    manifest = dict(status="completed", metadata=dict(input_sha256="same", expected_output_sha256="gold",
        judge_mode="llm", judge_model="pinned", jev_model=None, prompt_hashes={}, code_hashes={}),
        metrics=dict(missing_count=2, extra_count=3, classification_error_count=1, granularity_error_count=0,
                     field_scores={"formula": None, "units": .5}))
    baseline = write_json(tmp_path / "baseline.json", manifest)
    candidate = write_json(tmp_path / "candidate.json", manifest)
    assert compare_runs(baseline, candidate)["fields"]["formula"]["delta"] is None
    manifest["metadata"]["expected_output_sha256"] = "changed"
    write_json(candidate, manifest)
    with pytest.raises(ValueError, match="expected_output_sha256"):
        compare_runs(baseline, candidate)
