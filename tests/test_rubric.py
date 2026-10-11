import json
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from shop_agent_langgraph.evaluation import rubric as rb

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = ["RS-1", "RS-2", "RS-3", "AG-1", "AG-3", "AG-6", "AG-7"]


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
    assert [r.id for r in rb.load_rubrics().rubrics] == EXPECTED_IDS


@pytest.mark.parametrize("edit,message", [
    (lambda r: r.update(severity="veto"), "severity"),
    (lambda r: r.pop("question"), "question"),
    (lambda r: r.update(id="RS_1"), "invalid rubric id"),
])
def test_rubric_schema_violations_are_rejected(tmp_path, edit, message):
    directory = edited_dir(tmp_path, "research", lambda d: edit(d["rubrics"][0]))
    with pytest.raises((ValidationError, ValueError)) as error:
        rb.load_rubrics(directory)
    assert message in str(error.value)


def test_duplicate_wrong_prefix_and_missing_files_are_rejected(tmp_path):
    duplicate = edited_dir(tmp_path, "research", lambda d: d["rubrics"][1].update(id="RS-1"))
    with pytest.raises(ValidationError, match="duplicate"):
        rb.load_rubrics(duplicate)
    prefix = edited_dir(tmp_path / "b", "research", lambda d: d["rubrics"][0].update(id="XX-1"))
    with pytest.raises(ValidationError, match="must start with RS-"):
        rb.load_rubrics(prefix)
    missing = tmp_path / "c"
    shutil.copytree(rb.RUBRIC_DIR, missing)
    (missing / "research.yaml").unlink()
    with pytest.raises(ValueError, match="missing rubric file"):
        rb.load_rubrics(missing)
    swapped = tmp_path / "d"
    shutil.copytree(rb.RUBRIC_DIR, swapped)
    (swapped / "research.yaml").write_text((swapped / "aggregation.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="declares module"):
        rb.load_rubrics(swapped)
    extra = tmp_path / "e"
    shutil.copytree(rb.RUBRIC_DIR, extra)
    (extra / "route.yaml").write_text("module: route", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown rubric files"):
        rb.load_rubrics(extra)


def test_gamepad_overlay_matches_gold_and_is_a_draft():
    overlay = rb.load_overlay(ROOT / "eval/cases/gamepad/rubric.yaml")
    gold = json.loads((ROOT / "eval/cases/gamepad/gold.json").read_text(encoding="utf-8"))
    gold_ids = {item["id"] for kind in ("criteria", "attributes") for item in gold[kind]}
    assert overlay.status == "draft_pending_human_review"
    assert len(overlay.core_ids) == 18 and set(overlay.core_ids) <= gold_ids
    assert "input_latency" not in gold_ids
    assert "transmission_latency" not in gold_ids
    assert "价格" in overlay.excluded_keywords and len(overlay.excluded_keywords) == 10


def test_rubric_file_hashes_change_with_content(tmp_path):
    before = rb.rubric_file_hashes()
    assert set(before) == {f"evaluation/rubrics/{m}.yaml" for m in rb.MODULE_PREFIXES}
    directory = edited_dir(tmp_path, "research", lambda d: d["rubrics"][0].update(question="改过的问题"))
    after = rb.rubric_file_hashes(directory)
    assert [k for k in before if before[k] != after[k]] == ["evaluation/rubrics/research.yaml"]


# --- pass-rate summary ----------------------------------------------------------------

RUBRICS = rb.load_rubrics()


def units(*statuses):
    return [rb.UnitResult(unit=f"u{i}", status=s, reason="r") for i, s in enumerate(statuses)]


def test_pass_rate_counts_unresolved_against_and_skips_na():
    [o] = rb.summarize(RUBRICS, {"RS-1": units("pass", "pass", "fail", "unresolved", "na")})
    assert o.pass_rate == 0.5 and o.counts == {"pass": 2, "fail": 1, "unresolved": 1, "na": 1}
    assert o.failures == ["u2: r", "u3: r"]


def test_summary_has_no_rate_without_judged_units_and_skips_rubrics_not_run():
    outcomes = rb.summarize(RUBRICS, {"AG-1": units("na"), "RS-2": units("pass")})
    assert [(o.id, o.pass_rate) for o in outcomes] == [("RS-2", 1.0), ("AG-1", None)]


def test_markdown_renders_every_rate_and_failure():
    outcomes = rb.summarize(RUBRICS, {"RS-1": units("pass", "fail"), "RS-2": units("pass")})
    markdown = rb.render_markdown(outcomes, RUBRICS)
    assert "# Rubric 通过率" in markdown and "| RS-1 |" in markdown and "0.500" in markdown and "u1: r" in markdown


def test_rubric_doc_lists_exactly_the_shipped_rubrics():
    text = (ROOT / "docs/rubric.md").read_text(encoding="utf-8")
    listed = [line.split("|")[1].strip() for line in text.splitlines() if line.startswith(("| RS-", "| AG-"))]
    assert listed == EXPECTED_IDS
    for item in RUBRICS.rubrics:
        assert item.question in text, item.id
