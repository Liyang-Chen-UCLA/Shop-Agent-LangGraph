import json
import shutil
from pathlib import Path

import pytest

from shop_agent_langgraph.evaluation import rubric_checks as rc
from shop_agent_langgraph.evaluation import tasks as tk
from shop_agent_langgraph.domain.taxonomy import get_nodes

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_task_counts_and_draft_status():
    counts = {m: len(tk.load_tasks(m)) for m in ("intent", "route", "e2e", "research")}
    assert counts == {"intent": 5, "route": 5, "e2e": 3, "research": 9}
    assert {t.reference["action"] for t in tk.load_tasks("intent")} == {"create", "update", "remove", "query", "switch"}
    routes = tk.load_tasks("route")
    assert {t.reference["status"] for t in routes} == {"resolved", "ambiguous"}
    assert all(not tk.gold_approved(t) for module in counts for t in tk.load_tasks(module))


def test_reference_node_ids_exist_in_the_taxonomy():
    ids = set()
    for task in tk.load_tasks("route") + tk.load_tasks("e2e"):
        ref = task.reference
        ids |= {str(x) for x in [ref.get("node_id"), *ref.get("acceptable_node_ids", []), *ref.get("allowed_node_ids", []),
                                  *ref.get("candidate_node_ids", [])] if x}
    found, missing = get_nodes(sorted(ids))
    assert not missing and len(found) == len(ids)


def test_research_tasks_reference_gold_without_copying_it():
    gold = json.loads((ROOT / "eval/cases/gamepad/gold.json").read_text(encoding="utf-8"))
    tasks = tk.load_tasks("research")
    assert sorted(t.reference["item_id"] for t in tasks) == sorted(gold["source_item_ids"])
    for task in tasks:
        text = json.dumps([task.input, task.reference, task.metadata], ensure_ascii=False)
        assert "quote" not in text and "direction" not in text and len(text) < 600
    sample = tk.research_sample(tasks[0], {"item_id": tasks[0].id, "relevance": "relevant", "evidence": []}, context="原文")
    assert sample["reference"]["gold_evidence"] and all({"dimension_id", "quote"} <= set(e) for e in sample["reference"]["gold_evidence"])


def test_task_rubric_ids_must_belong_to_the_module(tmp_path):
    shutil.copytree(tk.TASKS_DIR / "route", tmp_path / "route")
    path = next((tmp_path / "route").glob("*.json"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rubric_ids"] = ["INT-1"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct route rubrics"):
        tk.load_tasks("route", directory=tmp_path / "route")
    raw["rubric_ids"] = ["RT-1"]
    raw["extra"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly input"):
        tk.load_tasks("route", directory=tmp_path / "route")


def test_every_task_rubric_runs_against_a_matching_fake_output():
    intent = tk.load_tasks("intent")[0]
    out = {"action": "create", "category": "游戏手柄", "criteria_preferences": ["价格 ≤ 500 元"], "attribute_preferences": []}
    results = rc.run_module("intent", {intent.id: tk.intent_sample(intent, out)}, only=set(intent.rubric_ids))
    assert all(u.status == "pass" for r in results.values() for u in r)
