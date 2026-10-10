import json
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from shop_agent_langgraph.evaluation import rubric as rb

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = ([f"E2E-{i}" for i in range(1, 8)] + [f"INT-{i}" for i in range(1, 6)]
                + [f"RT-{i}" for i in range(1, 6)] + [f"MS-{i}" for i in range(1, 6)]
                + [f"RS-{i}" for i in range(1, 6)] + [f"AG-{i}" for i in range(1, 8)])


def edited_dir(tmp_path, module, edit):
    """Copy the rubric assets and apply `edit(parsed_yaml)` to one module file."""
    target = tmp_path / "rubrics"
    shutil.copytree(rb.RUBRIC_DIR, target)
    path = target / f"{module}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return target


# --- YAML assets and validation -------------------------------------------------

def test_shipped_rubrics_load_with_the_specified_ids():
    rubrics = rb.load_rubrics()
    assert sorted(rubrics.by_id) == sorted(EXPECTED_IDS)
    for item in rubrics.rubrics:
        assert (item.judge == "llm") == (item.maturity == "shadow")
        assert (item.severity == "gate") == (item.threshold is not None)
    assert {r.id for r in rubrics.rubrics if r.judge == "llm"} == {"E2E-5", "INT-3", "INT-4", "RS-4", "RS-5", "AG-5"}
    assert rubrics.by_id["AG-3"].threshold == 0.95 and rubrics.by_id["E2E-6"].threshold == 1.0
    assert {r.id for r in rubrics.rubrics if r.depends_on_gold} == {
        "E2E-3", "INT-1", "INT-2", "INT-4", "RT-1", "RT-2", "RT-4", "MS-3", "MS-5", "RS-5",
        "AG-1", "AG-2", "AG-3", "AG-4", "AG-5"}


@pytest.mark.parametrize("module,rubric_id,edit,message", [
    ("intent", "INT-3", lambda r: r.update(maturity="active"), "llm rubrics must start as shadow"),
    ("intent", "INT-1", lambda r: r.update(maturity="shadow"), "code rubrics must be active"),
    ("intent", "INT-1", lambda r: r.pop("threshold"), "need a threshold"),
    ("intent", "INT-1", lambda r: r.update(threshold=1.5), "threshold in"),
    ("intent", "INT-5", lambda r: r.update(threshold=0.9), "only gate rubrics"),
    ("intent", "INT-1", lambda r: r.update(judge="human"), "judge"),
    ("intent", "INT-1", lambda r: r.pop("question"), "question"),
    ("intent", "INT-1", lambda r: r.update(unknown=1), "unknown"),
])
def test_rubric_policy_violations_are_rejected(tmp_path, module, rubric_id, edit, message):
    def apply(data):
        edit(next(r for r in data["rubrics"] if r["id"] == rubric_id))

    directory = edited_dir(tmp_path, module, apply)
    with pytest.raises((ValidationError, ValueError)) as error:
        rb.load_rubrics(directory)
    assert message in str(error.value)


def test_duplicate_wrong_prefix_and_missing_files_are_rejected(tmp_path):
    duplicate = edited_dir(tmp_path, "route", lambda d: d["rubrics"][1].update(id="RT-1"))
    with pytest.raises(ValidationError, match="duplicate"):
        rb.load_rubrics(duplicate)
    prefix = edited_dir(tmp_path / "b", "route", lambda d: d["rubrics"][0].update(id="XX-1"))
    with pytest.raises(ValidationError, match="must start with RT-"):
        rb.load_rubrics(prefix)
    missing = tmp_path / "c"
    shutil.copytree(rb.RUBRIC_DIR, missing)
    (missing / "route.yaml").unlink()
    with pytest.raises(ValueError, match="missing rubric file"):
        rb.load_rubrics(missing)
    swapped = tmp_path / "d"
    shutil.copytree(rb.RUBRIC_DIR, swapped)
    (swapped / "route.yaml").write_text((swapped / "intent.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="declares module"):
        rb.load_rubrics(swapped)


def test_gamepad_overlay_matches_gold_and_is_a_draft():
    overlay = rb.load_overlay(ROOT / "eval/cases/gamepad/rubric.yaml")
    gold = json.loads((ROOT / "eval/cases/gamepad/gold.json").read_text(encoding="utf-8"))
    gold_ids = {item["id"] for kind in ("criteria", "attributes") for item in gold[kind]}
    assert overlay.status == "draft_pending_human_review" and not overlay.approved
    assert len(overlay.core_ids) == 18 and set(overlay.core_ids) <= gold_ids
    assert "input_latency" not in gold_ids
    assert "transmission_latency" not in gold_ids
    assert "价格" in overlay.excluded_keywords and len(overlay.excluded_keywords) == 10


def test_rubric_file_hashes_change_with_content(tmp_path):
    before = rb.rubric_file_hashes()
    assert set(before) == {f"evaluation/rubrics/{m}.yaml" for m in rb.MODULE_PREFIXES}
    directory = edited_dir(tmp_path, "route", lambda d: d["rubrics"][0].update(question="改过的问题"))
    after = rb.rubric_file_hashes(directory)
    assert [k for k in before if before[k] != after[k]] == ["evaluation/rubrics/route.yaml"]


# --- gate ------------------------------------------------------------------------------

RUBRICS = rb.load_rubrics()


def units(*statuses):
    return [rb.UnitResult(unit=f"u{i}", status=s, reason="r") for i, s in enumerate(statuses)]


def outcome(rubric_id, *trials, approved=True):
    return rb.evaluate_rubric(RUBRICS.by_id[rubric_id], [units(*t) for t in trials], gold_approved=approved)


def gate(trials, **kwargs):
    return rb.evaluate_gate(RUBRICS, [{k: units(*v) for k, v in t.items()} for t in trials], **kwargs)


def test_active_veto_failure_blocks_and_gate_uses_threshold():
    assert outcome("RS-1", ["pass", "pass", "fail"]).verdict == "block"
    assert outcome("RS-1", ["pass", "pass", "na"]).verdict == "pass"
    assert outcome("E2E-4", ["pass"] * 9 + ["fail"]).verdict == "pass"       # 0.9 meets the 0.9 threshold
    low = outcome("E2E-4", ["pass"] * 8 + ["fail"] * 2)
    assert low.verdict == "block" and low.pass_rate == 0.8 and "threshold 0.9" in low.reason


def test_shadow_and_monitor_rubrics_only_report():
    shadow = outcome("E2E-5", ["fail", "fail"])               # llm veto, shadow
    assert shadow.verdict == "report" and shadow.would_be == "block"
    monitor = outcome("AG-2", ["fail", "pass"])
    assert monitor.verdict == "report" and monitor.pass_rate == 0.5
    report = gate([{"E2E-5": ["fail"], "AG-2": ["fail"], "E2E-4": ["pass"]}])
    assert report.conclusion == "pass" and not report.blockers


def test_gold_dependent_failures_are_downgraded_until_gold_is_approved():
    unapproved = outcome("AG-1", ["pass", "fail"], approved=False)
    assert unapproved.verdict == "needs_review" and unapproved.gold_downgraded
    assert outcome("AG-1", ["pass", "fail"], approved=True).verdict == "block"
    independent = outcome("INT-5", ["fail"], approved=False)             # not gold-dependent
    assert independent.verdict == "block" and not independent.gold_downgraded
    assert outcome("AG-1", ["pass"], approved=False).verdict == "pass"
    assert gate([{"AG-1": ["fail"]}], gold_approved=False).conclusion == "needs_review"
    assert gate([{"AG-1": ["fail"]}], gold_approved=True).conclusion == "block"
    assert gate([{"AG-1": ["fail"]}], gold_approved={"aggregation": True}).conclusion == "block"


def test_conclusion_priority_is_block_then_needs_review_then_pass():
    mixed = gate([{"AG-1": ["fail"], "AG-7": ["fail"]}], gold_approved=False)
    assert mixed.conclusion == "block" and mixed.blockers == ["AG-7"] and mixed.reviews == ["AG-1"]
    assert gate([{"AG-1": ["fail"], "AG-7": ["pass"]}], gold_approved=False).conclusion == "needs_review"
    assert gate([{"AG-1": ["pass"], "AG-7": ["pass"]}], gold_approved=False).conclusion == "pass"


def test_unresolved_units_never_silently_pass():
    assert outcome("RS-1", ["pass", "unresolved"]).verdict == "needs_review"
    assert outcome("RS-1", ["pass", "unresolved", "fail"]).verdict == "block"
    reachable = outcome("E2E-4", ["pass"] * 8 + ["unresolved"] * 2)       # 0.8 now, 1.0 if resolved
    assert reachable.verdict == "needs_review" and reachable.pass_rate == 0.8
    assert outcome("E2E-4", ["pass"] * 5 + ["fail"] * 3 + ["unresolved"] * 2).verdict == "block"


def test_trials_veto_needs_every_trial_clean_and_gate_averages():
    assert outcome("RS-1", ["pass"], ["pass"], ["pass"]).verdict == "pass"
    assert outcome("RS-1", ["pass"] * 9, ["pass"] * 8 + ["fail"]).verdict == "block"   # one bad trial is enough
    averaged = outcome("E2E-4", ["pass"] * 10, ["pass"] * 8 + ["fail"] * 2)            # mean(1.0, 0.8) = 0.9
    assert averaged.verdict == "pass" and averaged.pass_rate == pytest.approx(0.9) and averaged.trials == 2
    assert outcome("E2E-4", ["pass"] * 10, ["pass"] * 7 + ["fail"] * 3).verdict == "block"
    pooled_would_differ = outcome("E2E-4", ["pass"], ["pass"] * 2 + ["fail"] * 8)      # mean 0.6, pooled 0.27 irrelevant
    assert pooled_would_differ.pass_rate == pytest.approx(0.6)
    assert outcome("E2E-4", ["na"], ["pass"]).verdict == "pass"                        # empty trial is skipped


def test_not_run_and_not_applicable_are_distinguished():
    assert rb.evaluate_rubric(RUBRICS.by_id["RS-4"], [], gold_approved=True).verdict == "not_run"
    assert outcome("RS-1", ["na"]).verdict == "na"
    report = gate([{"RS-1": ["pass"]}])
    assert {o.id: o.verdict for o in report.outcomes}["RS-5"] == "not_run" and report.conclusion == "pass"


def test_gate_report_renders_every_rubric_and_failures():
    report = gate([{"RS-1": ["pass", "fail"], "RS-2": ["pass"]}])
    markdown = rb.render_gate_markdown(report)
    assert "# Rubric 门禁：block" in markdown and "| RS-1 |" in markdown and "u1: r" in markdown
    assert report.model_dump(mode="json")["blockers"] == ["RS-1"]


# --- promotion and documentation -----------------------------------------------------------

def entry(count, correct, positive, negative, unclear=0):
    return dict(count=count, correct=correct, unclear=unclear, positive_labels=positive, negative_labels=negative)


def test_promotion_needs_agreement_samples_and_both_label_classes():
    assert rb.promotion_status(entry(30, 27, 20, 10))["eligible"]                       # exactly 0.9 and 30
    low = rb.promotion_status(entry(40, 35, 20, 20))
    assert not low["eligible"] and "agreement 0.875" in low["reasons"][0]
    assert "29 human samples" in rb.promotion_status(entry(29, 29, 20, 9))["reasons"][0]
    one_sided = rb.promotion_status(entry(50, 50, 50, 0))
    assert one_sided["agreement"] == 1.0 and one_sided["reasons"] == ["human labels lack positive or negative examples"]
    assert not rb.promotion_status(entry(0, 0, 0, 0))["eligible"]


def test_rubric_doc_lists_every_rubric_with_its_level_and_threshold():
    text = (ROOT / "docs/rubric.md").read_text(encoding="utf-8")
    for item in rb.load_rubrics().rubrics:
        row = next((line for line in text.splitlines() if line.startswith(f"| {item.id} |")), None)
        assert row is not None, item.id
        cells = [c.strip() for c in row.strip("|").split("|")]
        level = item.severity + (f" {item.threshold}" if item.threshold else "")
        assert cells[2:5] == [item.judge, level, item.maturity] and cells[5] == ("是" if item.depends_on_gold else "否"), item.id
    for number in ("0.9", "30"):
        assert number in text.split("## shadow → active 晋升")[1]
