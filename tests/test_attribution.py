import json
from contextlib import contextmanager

import pytest

from shop_agent_langgraph.evaluation import attribution as at
from shop_agent_langgraph.evaluation import failure_clusters as fc
from test_failure_clusters import FakeModel
from test_rubric_integration import replay

AI_REASON = "AI 的独有理由文本"


@pytest.fixture
def clustered(tmp_path):
    run = replay(tmp_path, "run")
    rubric = json.loads((run / "rubric.json").read_text(encoding="utf-8"))
    rubric["trial"]["AG-6"].append({"unit": "delivery/warranty", "status": "fail", "reason": "excluded keywords: ['质保']"})
    rubric["trial"]["AG-5"].append({"unit": "delivery/runtime.units", "status": "fail",
                                    "reason": "Gold allows hours and minutes; actual only hours"})
    (run / "rubric.json").write_text(json.dumps(rubric, ensure_ascii=False), encoding="utf-8")
    return fc.cluster_failures([run], FakeModel())


class OwnerModel:
    """Owns the cluster as agent and marks the injected AG-5 unit as a gold exception."""

    def with_structured_output(self, schema, method):
        class Bound:
            def invoke(self, messages):
                payload = json.loads(messages[1]["content"])
                assert "ai_owner" not in json.dumps(payload)
                key = next(k for k, u in payload["units"].items() if u["rubric"] == "AG-5")
                return schema(ai_owner="agent", reason=AI_REASON,
                              exceptions=[dict(unit=key, ai_owner="gold", reason="gold 单位不全")])
        return Bound()


def test_tasks_split_exceptions_and_hide_ai_owner(clustered):
    owners = at.propose_owners(clustered, OwnerModel())
    [cluster] = clustered["clusters"]
    [exception] = owners[cluster["cluster_id"]]["exceptions"]
    assert exception["unit"]["unit"] == "delivery/runtime.units"

    tasks, ai = at.attribution_tasks(clustered, owners)
    assert [t["task_kind"] for t in tasks] == ["cluster", "unit"]
    assert tasks[1]["units"] == [exception["unit"]] and exception["unit"] not in tasks[0]["units"]
    assert {ai[t["id"]]["ai_owner"] for t in tasks} == {"agent", "gold"}
    visible = json.dumps([{k: t[k] for k in ("input", "output", "metadata")} for t in tasks], ensure_ascii=False)
    assert AI_REASON not in visible and "ai_owner" not in visible and "gold 单位不全" not in visible
    assert list(tasks[0]["input"])[:4] == ["问题", "这一簇", "规则说明", "为什么判为失败"]
    assert list(tasks[0]["input"]["负责方定义"]) == [at.OWNER_LABELS[o] for o in at.OWNERS]
    example = tasks[0]["input"]["代表样例"][0]
    assert {"单元", "失败原因"} <= set(example) and all(not k.isascii() or k == "gold 维度" for k in example if k != "单元")
    assert set(tasks[0]["output"]) == {"涉及规则", "涉及商品", "涉及维度", "失败单元"}
    [why] = tasks[1]["input"]["为什么判为失败"]
    assert "units" in why["说明"] and why["检查器原始理由"].startswith("Gold allows")
    assert [t["id"] for t in at.attribution_tasks(clustered, owners)[0]] == [t["id"] for t in tasks]   # stable


def test_publishing_twice_uses_the_shared_queue_once(monkeypatch, tmp_path, clustered):
    import shop_agent_langgraph.evaluation.annotations as annotations
    tasks, _ = at.attribution_tasks(clustered, at.propose_owners(clustered, OwnerModel()))
    queue = {"id": "queue", "name": annotations.QUEUE_NAME, "scoreConfigIds": ["human_correct"]}
    created, items, names = [], [], []

    def list_items(resource, action, *args):
        if resource == "score-configs":
            return created
        return [queue] if action == "list" else items

    def api(resource, action, *args, body=None):
        if action == "create-item":
            items.append(body)
            return body
        created.append({**body, "id": body["name"]})
        return created[-1]

    class Client:
        @contextmanager
        def start_as_current_observation(self, **kwargs):
            names.append(kwargs["name"])
            yield type("Span", (), {"trace_id": "trace", "id": f"observation-{len(names)}"})()

        def flush(self):
            pass

    for module in (annotations, at):
        monkeypatch.setattr(module, "all_items", list_items)
    monkeypatch.setattr(annotations, "api", api)
    with pytest.raises(ValueError, match="Langfuse UI"):            # config not yet on the queue
        at.ensure_attribution_queue()
    [config] = created
    assert config["dataType"] == "CATEGORICAL" and config["name"] == "失败负责方"
    assert [c["label"] for c in config["categories"]] == [at.OWNER_LABELS[o] for o in at.OWNERS]
    queue["scoreConfigIds"].append(config["id"])
    for _ in range(2):
        annotations.publish_tasks(tasks, client=Client(), ledger_path=tmp_path / "ledger.json",
                                  queue=at.ensure_attribution_queue())
    assert names == ["归因-失败簇", "归因-单个失败单元"] and len(items) == len(tasks)


def reviewed(tasks, owners):
    return [dict(task_id=t["id"], task=t, human_score=o, comment="", reviewer="u", trace_id="t", observation_id=f"o{i}")
            for i, (t, o) in enumerate(zip(tasks, owners))]


def test_summary_and_followups_follow_the_human_label(clustered):
    tasks, ai = at.attribution_tasks(clustered, at.propose_owners(clustered, OwnerModel()))
    unit_task = next(t for t in tasks if t["task_kind"] == "unit")

    summary = at.summarize(reviewed(tasks, ["agent", "judge"]), ai)
    assert summary["count"] == 2 and summary["agreement"] == 0.5
    assert {"human": "judge", "ai": "gold", "count": 1} in summary["confusion"]

    # agent/rubric never create scoring tasks; only the human label decides, not the AI owner.
    assert at.followup_tasks(reviewed(tasks, ["agent", "rubric"])) == []
    [judge] = at.followup_tasks(reviewed([unit_task], ["judge"]))
    assert (judge["task_kind"], judge["field"]) == ("schema_field", "units")
    assert judge["metadata"]["from_attribution_task"] == unit_task["id"]
    [gold] = at.followup_tasks(reviewed([unit_task], ["gold"]))
    assert (gold["task_kind"], gold["ref"]) == ("gold_item", "runtime")


def test_ai_scores_are_attached_with_stable_ids(monkeypatch, clustered):
    tasks, ai = at.attribution_tasks(clustered, at.propose_owners(clustered, OwnerModel()))
    scores = []

    class Client:
        def create_score(self, **kwargs):
            scores.append(kwargs)

        def flush(self):
            pass

    monkeypatch.setattr(at, "ensure_score_config", lambda *a, **k: {"id": "ai-config"})
    at.attach_ai_scores(reviewed(tasks, ["agent", "gold"]), ai, Client())
    at.attach_ai_scores(reviewed(tasks, ["agent", "gold"]), ai, Client())
    assert [s["value"] for s in scores[:2]] == [at.OWNER_LABELS[ai[t["id"]]["ai_owner"]] for t in tasks]
    assert at.owner_codes([{"human_score": "gold 有误"}]) == [{"human_score": "gold"}]
    assert scores[0]["score_id"] == scores[2]["score_id"] and scores[0]["config_id"] == "ai-config"


def test_rules_and_failures_are_explained_in_plain_words():
    from shop_agent_langgraph.evaluation.rubric import load_overlay, load_rubrics
    from shop_agent_langgraph.evaluation.rubric_run import DEFAULT_OVERLAY
    rule = at.rule_explanation("AG-6", load_rubrics(), load_overlay(DEFAULT_OVERLAY))
    assert "质保" in rule["排除关键词"] and rule["级别"].startswith("一票否决") and rule["是否依赖 gold"].startswith("否")
    assert "0.95" in at.rule_explanation("AG-3", load_rubrics(), load_overlay(DEFAULT_OVERLAY))["级别"]

    def explain(rubric, unit, reason, **extra):
        obj, field = fc.split_unit(rubric, unit)
        return at.explain_unit(dict(rubric=rubric, object=obj, field=field, reason=reason, **extra))

    assert "排除关键词 '质保'" in explain("AG-6", "delivery/warranty", "excluded keywords: ['质保']", name="质保时长")
    assert "「质保时长」" in explain("AG-6", "delivery/warranty", "excluded keywords: ['质保']", name="质保时长")
    kind = explain("AG-3", "delivery/bt.kind", "gold is attribute, actual is ['criterion']")
    assert "attributes（分类属性" in kind and "criteria（质量标准" in kind
    assert "合并成了一个" in explain("AG-4", "delivery/motor", "3 gold to 1 actual")
    assert "拆成了多个" in explain("AG-4", "delivery/conn", "1 gold to 2 actual")
    assert "只做监控" in explain("AG-2", "delivery/x", "dimension is missing from Market")
    assert "冻结原文里找不到" in explain("RS-1", "p1/e3", "source_text must be a substring of the product text")
    assert "单位只给了小时" in explain("AG-5", "delivery/t.units", "Gold allows hours", note="单位只给了小时")
