import json

import pytest

from shop_agent_langgraph.evaluation import failure_clusters as fc
from test_rubric_integration import replay


def unit(rubric, name, reason, run=0, status="fail"):
    obj, field = fc.split_unit(rubric, name)
    signature = fc.template(reason, field) or reason
    return dict(id=f"r{run}|{rubric}|{name}", run=run, rubric=rubric, unit=name, object=obj, field=field,
                status=status, reason=reason, signature=signature, root=None)


REPORT = {"items": {"gold:0": {"item": {"id": "motor_type"}}, "gold:1": {"item": {"id": "motor_count"}},
                    "actual:0": {"item": {"id": "motor"}}},
          "match": [{"gold_ids": ["gold:0", "gold:1"], "actual_ids": ["actual:0"], "reason": "合并"}]}


def test_signatures_drop_specific_values():
    assert fc.template("gold is attribute, actual is ['criterion']", "kind") == "kind: attribute→criterion"
    assert fc.template("3 gold to 1 actual", "") == "多个 gold 合并为 1 个输出"
    assert fc.template("excluded keywords: ['质保']", "") == fc.template("excluded keywords: ['销量']", "")
    assert fc.template("core dimension is missing from Market", "") == "核心维度缺失"
    assert fc.template("something new", "units") is None                   # falls back to the raw reason
    assert fc.research_feature("RS-3", {"value": "≤8", "source_text": "距离： $ \\leq8M $"}) == "引文含 LaTeX/公式写法"
    assert fc.research_feature("RS-3", {"value": "PC/安卓", "source_text": "PC 电脑 安卓"}) == "value 由多处原文拼接"


def test_cascades_attach_to_merge_or_kind_type_root():
    units = [
        unit("AG-3", "delivery/grip.kind", "gold is attribute, actual is ['criterion']"),
        unit("AG-3", "delivery/grip.direction", "field exists on actual only"),
        unit("AG-1", "delivery/motor_type", "2 gold to 1 actual"),
        unit("AG-3", "delivery/motor_count.kind", "gold is attribute, actual is ['criterion']"),
        unit("AG-3", "delivery/motor_count.direction", "field exists on actual only"),
        unit("AG-3", "delivery/battery.units", "field exists on gold only"),
    ]
    fc.attach_cascades(units, [{"report": REPORT}])
    roots = {u["id"]: u["root"] for u in units}
    assert roots["r0|AG-3|delivery/grip.direction"] == "r0|AG-3|delivery/grip.kind"
    merge = "r0|AG-1|delivery/motor_type"                          # same matching group
    assert roots["r0|AG-3|delivery/motor_count.kind"] == roots["r0|AG-3|delivery/motor_count.direction"] == merge
    assert roots["r0|AG-3|delivery/battery.units"] is None and roots[merge] is None
    groups = fc.initial_groups(units)
    assert sorted(len(g["units"]) for g in groups) == [1, 1, 1]


def test_initial_groups_pool_runs_by_signature():
    units = [unit("AG-6", "delivery/warranty", "excluded keywords: ['质保']", run=0),
             unit("AG-6", "delivery/sales", "excluded keywords: ['销量']", run=1)]
    [group] = fc.initial_groups(units)
    assert group["signature"] == "输出含排除类关键词" and len(group["units"]) == 2


def test_proposal_must_cover_every_root():
    groups = [dict(id="g1", rubric="AG-6", signature="s", units=["a", "b"]), dict(id="g2", rubric="AG-1", signature="t", units=["c"])]
    ok = fc.ClusterProposal(clusters=[fc.ProposedCluster(name="x", groups=["g1"], units=["c"], hypothesis="h", evidence=["a"])])
    fc.check_proposal(ok, groups)
    split = fc.ClusterProposal(clusters=[fc.ProposedCluster(name="x", groups=["g1", "g2"], hypothesis="h", evidence=["a"]),
                                         fc.ProposedCluster(name="y", units=["b"], hypothesis="h", evidence=["b"])])
    fc.check_proposal(split, groups)                 # b is taken out of g1 by being listed on its own
    assert fc.expand(split, groups) == [["a", "c"], ["b"]]
    twice = fc.ClusterProposal(clusters=[fc.ProposedCluster(name="x", groups=["g1"], units=["c"], hypothesis="h", evidence=[]),
                                         fc.ProposedCluster(name="y", units=["c", "b"], hypothesis="h", evidence=[])])
    assert [sorted(u) for u in fc.expand(twice, groups)] == [["a", "c"], ["b"]]   # c stays with x; b leaves g1
    shared = fc.ClusterProposal(clusters=[fc.ProposedCluster(name="x", groups=["g1", "g2"], hypothesis="h", evidence=[]),
                                          fc.ProposedCluster(name="y", groups=["g2"], hypothesis="h", evidence=[])])
    assert fc.expand(shared, groups) == [["a", "b", "c"], []]      # a group claimed twice stays with the first
    for bad in ([fc.ProposedCluster(name="x", groups=["g1"], hypothesis="h", evidence=["a"])],              # omits c
                [fc.ProposedCluster(name="x", groups=["g1", "g2", "g9"], hypothesis="h", evidence=[])]):     # unknown group
        with pytest.raises(ValueError):
            fc.check_proposal(fc.ClusterProposal(clusters=bad), groups)


class FakeModel:
    """Merges every initial group into one cluster; the first answer omits a group to exercise the retry."""

    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema, method):
        model = self

        class Bound:
            def invoke(self, messages):
                model.calls.append((schema.__name__, messages))
                payload = json.loads(messages[1]["content"])
                ids = [g["id"] for g in payload["groups"]]
                first = sum(1 for name, _ in model.calls if name == "ClusterProposal") == 1
                chosen = ids[:-1] if first and len(ids) > 1 else ids
                return schema(clusters=[dict(name="全部", groups=chosen, hypothesis="如果改 X，这些失败会一起消失",
                                             evidence=[payload["groups"][0]["units"][0]["id"]])])
        return Bound()


def test_cluster_failures_on_saved_run(tmp_path):
    run = replay(tmp_path, "run")
    rubric = json.loads((run / "rubric.json").read_text(encoding="utf-8"))
    # Inject failures the replay does not produce, so every stage has input.
    rubric["trial"]["AG-6"].append({"unit": "delivery/warranty", "status": "fail", "reason": "excluded keywords: ['质保']"})
    rubric["trial"]["AG-3"].append({"unit": "delivery/battery_life.kind", "status": "fail",
                                    "reason": "gold is criterion, actual is ['attribute']"})
    (run / "rubric.json").write_text(json.dumps(rubric, ensure_ascii=False), encoding="utf-8")

    model = FakeModel()
    result = fc.cluster_failures([run], model)
    fc.write_clusters(result, tmp_path / "out")

    assert len(result["groups"]) >= 2
    [cluster] = result["clusters"]
    assert cluster["changed_by_llm"] and set(cluster["merged_signatures"]) == {g["id"] for g in result["groups"]}
    assert cluster["cluster_id"] == cluster["merged_signatures"][0]          # largest initial group anchors the ID
    assert {"run_dir": run.as_posix(), "rubric_id": "AG-6", "unit": "delivery/warranty"} in cluster["units"]
    assert {"AG-3", "AG-6"} <= set(cluster["rubric_ids"]) and "warranty" in cluster["dimension_ids"]
    assert cluster["item_ids"] == [] and cluster["runs"] == [{"run_dir": run.as_posix(), "trace_id": None}]
    assert [name for name, _ in model.calls].count("ClusterProposal") == 2      # rejected once, then accepted
    assert "被拒绝" in model.calls[-1][1][-1]["content"]
    markdown = (tmp_path / "out" / "clusters.md").read_text(encoding="utf-8")
    assert "⚠ 全部" in markdown and "如果改 X" in markdown
    assert result["rubric_hashes"] == rubric["rubric_hashes"]


def test_cluster_ids_depend_only_on_signatures():
    first = [unit("AG-6", "delivery/warranty", "excluded keywords: ['质保']", run=0)]
    second = [unit("AG-6", "delivery/sales", "excluded keywords: ['销量']", run=3),
              unit("AG-1", "delivery/vibration", "core dimension is missing from Market", run=3)]
    ids = lambda units: {g["id"] for g in fc.initial_groups(units)}
    assert ids(first) == {"AG-6:输出含排除类关键词"} and ids(first) <= ids(second)


def test_samples_show_the_gold_side_with_its_evidence():
    report = {"items": {"gold:0": {"kind": "attribute", "item": {"id": "bt", "type": "categorical"}},
                        "actual:0": {"kind": "criterion", "item": {"id": "bt", "type": "numeric"}},
                        "gold:1": {"kind": "criterion", "item": {"id": "vibration"}}},
              "match": [{"gold_ids": ["gold:0"], "actual_ids": ["actual:0"], "reason": "同一维度"}],
              "missing": [{"ref": "gold:1", "reason": "未记录震动反馈"}], "extra": []}
    case = {"gold_annotation": {"criteria": [{"id": "vibration", "evidence": [{"item_id": "p1", "quote": "双马达震动", "page_id": "x"}]}],
                                "attributes": [{"id": "bt", "evidence": []}]}}
    contexts = [{"report": report, "case": case, "research": {}}]
    shared = fc.sample_context(unit("AG-3", "delivery/bt.type", "gold is categorical, actual is ['numeric']"), contexts)
    assert shared["gold"]["item"]["type"] == "categorical" and shared["matching"]["reason"] == "同一维度"
    missing = fc.sample_context(unit("AG-1", "delivery/vibration", "core dimension is missing from Market"), contexts)
    assert missing["gold_evidence"] == [{"item_id": "p1", "quote": "双马达震动"}]
    assert missing["matching"] == {"result": "missing", "reason": "未记录震动反馈"}
    assert "actual" in fc.sample_context(unit("AG-6", "delivery/bt", "excluded keywords: ['x']"), contexts)
