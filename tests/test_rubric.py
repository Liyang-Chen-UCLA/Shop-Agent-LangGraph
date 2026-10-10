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
    assert len(overlay.core_ids) == 20 and set(overlay.core_ids) <= gold_ids
    assert "价格" in overlay.excluded_keywords and len(overlay.excluded_keywords) == 10


def test_rubric_file_hashes_change_with_content(tmp_path):
    before = rb.rubric_file_hashes()
    assert set(before) == {f"evaluation/rubrics/{m}.yaml" for m in rb.MODULE_PREFIXES}
    directory = edited_dir(tmp_path, "route", lambda d: d["rubrics"][0].update(question="改过的问题"))
    after = rb.rubric_file_hashes(directory)
    assert [k for k in before if before[k] != after[k]] == ["evaluation/rubrics/route.yaml"]
