import json
import subprocess
import sys
from pathlib import Path

import pytest

from shop_agent_langgraph.evaluation import rubric as rb
from shop_agent_langgraph.evaluation.calibration import compare_runs
from shop_agent_langgraph.evaluation.cases import write_json
from shop_agent_langgraph.evaluation.experiment import run_experiment, versions
from shop_agent_langgraph.evaluation.rubric_run import gate_runs, rubric_scores
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


def test_experiment_saves_rubric_artifacts_and_manifest_gate(tmp_path):
    run = replay(tmp_path, "run")
    saved = json.loads((run / "rubric.json").read_text(encoding="utf-8"))
    assert {"AG-1", "AG-3", "AG-5", "AG-6", "AG-7"} <= set(saved["trial"]) and "RS-1" not in saved["trial"]  # replay: no research audit
    assert set(saved["rubric_hashes"]) == set(rb.rubric_file_hashes())
    assert saved["gold_approved"] is False and saved["gate"]["trial_count"] == 1
    assert "# Rubric 门禁" in (run / "rubric.md").read_text(encoding="utf-8")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rubric_gate"]["conclusion"] == saved["gate"]["conclusion"]
    assert "AG-7" not in manifest["rubric_gate"]["blockers"]       # unique ids and exact source IDs


def test_langfuse_scores_carry_each_rate_and_the_gate():
    report = rb.evaluate_gate(rb.load_rubrics(), [{"AG-7": [rb.UnitResult(unit="a", status="pass"),
                                                          rb.UnitResult(unit="b", status="fail")]}])
    scores = {s.name: s for s in rubric_scores(report)}
    assert scores["rubric.AG-7"].value == 0.5 and scores["rubric.AG-7"].data_type == "NUMERIC"
    assert scores["rubric.gate"].value == "block" and scores["rubric.gate"].data_type == "CATEGORICAL"
    assert "rubric.AG-5" not in scores                             # not run => no fabricated score


def test_versions_hash_every_rubric_file_and_overlay():
    hashes = versions()["prompt_hashes"]
    assert set(rb.rubric_file_hashes()) <= set(hashes) and "eval/cases/gamepad/rubric.yaml" in hashes


def test_compare_rejects_runs_scored_with_different_rubrics(tmp_path):
    def manifest(rubric_hash):
        return dict(status="completed", metadata=dict(input_sha256="s", expected_output_sha256="g", judge_mode="llm",
            judge_model="m", jev_model=None, code_hashes={},
            prompt_hashes={"evaluation/rubrics/route.yaml": rubric_hash}),
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


def test_gate_treats_several_runs_as_trials_and_rejects_mixed_rubrics(tmp_path):
    first, second = replay(tmp_path, "r1"), replay(tmp_path, "r2")
    report, runs = gate_runs([first, second])
    assert report.trial_count == 2 and [r["source"] for r in runs] == ["rubric.json"] * 2
    saved = json.loads((second / "rubric.json").read_text(encoding="utf-8"))
    saved["rubric_hashes"]["evaluation/rubrics/route.yaml"] = "changed"
    write_json(second / "rubric.json", saved)
    with pytest.raises(ValueError, match="different rubric files"):
        gate_runs([first, second])


def test_gate_recomputes_old_runs_and_cli_exits_one_on_block(tmp_path):
    run = replay(tmp_path, "old")
    (run / "rubric.json").unlink()
    market = json.loads((run / "market.json").read_text(encoding="utf-8"))
    market["item_ids"] = ["p1", "p1"]                                # AG-7 veto: source_item_ids differ from input
    write_json(run / "market.json", market)
    report, runs = gate_runs([run])
    assert runs[0]["source"] == "recomputed" and report.conclusion == "block" and "AG-7" in report.blockers
    done = subprocess.run([sys.executable, str(ROOT / "scripts/market_evaluation.py"), "gate", "--runs", str(run)],
                          capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert done.returncode == 1 and "GATE: block" in done.stdout
