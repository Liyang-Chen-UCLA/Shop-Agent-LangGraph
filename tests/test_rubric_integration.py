import json
from pathlib import Path

import pytest

from shop_agent_langgraph.evaluation import rubric as rb
from shop_agent_langgraph.evaluation.calibration import compare_runs
from shop_agent_langgraph.evaluation.cases import write_json
from shop_agent_langgraph.evaluation.experiment import run_experiment, versions
from shop_agent_langgraph.evaluation.rubric_run import load_runs, rubric_scores
from test_eval_agent import review
from test_fixed_market_evaluation import RecordingModel, call, small_case

ROOT = Path(__file__).resolve().parents[1]


def replay(tmp_path, name):
    case = small_case()
    actual = write_json(tmp_path / f"{name}-actual.json", case["expected_output"])
    model = RecordingModel(responses=[call("MatchingReport", dict(match=[dict(
        gold_ids=["gold:0"], actual_ids=["actual:0"], reason="续航")], missing=[], extra=[])),
        call("SchemaReview", {"items": [review(fields=("units", "direction", "aliases"))]})])
    run_experiment(case, tmp_path / name, actual_path=actual, model=model)
    return tmp_path / name


def test_experiment_saves_rubric_artifacts(tmp_path):
    run = replay(tmp_path, "run")
    saved = json.loads((run / "rubric.json").read_text(encoding="utf-8"))
    assert set(saved["trial"]) == {"AG-1", "AG-3", "AG-6", "AG-7"}      # replay: no research audit
    assert set(saved["rubric_hashes"]) == set(rb.rubric_file_hashes())
    assert {o["id"]: o["pass_rate"] for o in saved["summary"]}["AG-7"] == 1.0   # unique ids and exact source IDs
    assert "gate" not in saved and "# Rubric 通过率" in (run / "rubric.md").read_text(encoding="utf-8")
    assert "rubric_gate" not in json.loads((run / "manifest.json").read_text(encoding="utf-8"))


def test_langfuse_scores_carry_each_rate_only():
    outcomes = rb.summarize(rb.load_rubrics(), {"AG-7": [rb.UnitResult(unit="a", status="pass"),
                                                         rb.UnitResult(unit="b", status="fail")],
                                                "AG-6": [rb.UnitResult(unit="-", status="na")]})
    scores = {s.name: s for s in rubric_scores(outcomes)}
    assert set(scores) == {"rubric.AG-7"} and scores["rubric.AG-7"].value == 0.5
    assert scores["rubric.AG-7"].data_type == "NUMERIC"


def test_versions_hash_every_rubric_file_and_overlay():
    hashes = versions()["prompt_hashes"]
    assert set(rb.rubric_file_hashes()) | set(rb.schema_fields_hash()) <= set(hashes)
    assert "eval/cases/gamepad/rubric.yaml" in hashes


def test_compare_rejects_runs_scored_with_different_rubrics(tmp_path):
    def manifest(rubric_hash):
        return dict(status="completed", metadata=dict(input_sha256="s", expected_output_sha256="g", judge_mode="llm",
            judge_model="m", jev_model=None, code_hashes={},
            prompt_hashes={"evaluation/rubrics/research.yaml": rubric_hash}),
            metrics=dict(missing_count=0, extra_count=0, classification_error_count=0, granularity_error_count=0, field_scores={}))
    base = write_json(tmp_path / "b.json", manifest("a"))
    assert compare_runs(base, write_json(tmp_path / "c.json", manifest("a")))["counts"]
    with pytest.raises(ValueError, match="different rubric files"):
        compare_runs(base, write_json(tmp_path / "d.json", manifest("b")))
    code = manifest("a")
    code["metadata"]["code_hashes"] = {"evaluation/rubric_checks.py": "x"}
    other = manifest("a")
    other["metadata"]["code_hashes"] = {"evaluation/rubric_checks.py": "y"}
    with pytest.raises(ValueError, match="evaluation code"):
        compare_runs(write_json(tmp_path / "e.json", code), write_json(tmp_path / "f.json", other))


def test_load_runs_reads_saved_trials_and_recomputes_stale_or_missing_ones(tmp_path):
    first, second = replay(tmp_path, "r1"), replay(tmp_path, "r2")
    assert [r["source"] for r in load_runs([first, second])] == ["rubric.json"] * 2
    saved = json.loads((second / "rubric.json").read_text(encoding="utf-8"))
    saved["rubric_hashes"]["evaluation/rubrics/research.yaml"] = "changed"
    saved["trial"] = {"AG-4": []}                                    # scored with an older rubric set
    write_json(second / "rubric.json", saved)
    runs = load_runs([first, second])
    assert runs[1]["source"] == "recomputed" and "AG-4" not in runs[1]["trial"]
    (first / "rubric.json").unlink()
    market = json.loads((first / "market.json").read_text(encoding="utf-8"))
    market["item_ids"] = ["p1", "p1"]                                # source_item_ids differ from input
    write_json(first / "market.json", market)
    [run] = load_runs([first])
    assert run["source"] == "recomputed"
    assert {u.unit: u.status for u in run["trial"]["AG-7"]}["delivery/source_item_ids"] == "fail"
