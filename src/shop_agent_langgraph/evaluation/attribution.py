"""Human attribution of failure clusters in a Langfuse annotation queue.

Attribution tasks ask "who is responsible for this kind of failure" for one cluster (or one
exception unit) and record a CATEGORICAL `failure_owner`. The Hobby plan allows a single
annotation queue, so they share `market-dimension-review` with the scoring tasks; filter by
`task_kind` (cluster/unit) or the `归因-*` observation name. Everything a reviewer sees in
the queue (score names, category labels, task names and content) is in Chinese. The AI owner is
kept in a local file and attached as `ai_owner` only after the human label exists, so the
reviewer is not anchored and Score Analytics can compare both.

Clusters humans attribute to `gold` or `judge` become drafts for `market-dimension-review`;
gold changes only through that queue. The scoring queue never creates attribution tasks.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..agents.eval.schemas import EvalReport
from .annotations import QUEUE_NAME, all_items, ensure_score_config, review_tasks
from .cases import digest
from .failure_clusters import DERIVED_FROM_TYPE, split_unit, structured
from .rubric import GoldOverlay, RubricSet, load_overlay, load_rubrics
from .rubric_run import DEFAULT_OVERLAY

ROOT = Path(__file__).resolve().parents[3]
SCORE_NAME = "失败负责方"
AI_SCORE_NAME = "AI初判负责方"
ATTRIBUTION_VERSION = "v3"  # v3: Chinese task names, keys and category labels
OWNERS = ("agent", "rubric", "gold", "judge", "unclear")
# Category labels shown in the queue; codes stay English in files and code.
OWNER_LABELS = {"agent": "Agent 行为错误", "rubric": "规则或检查器问题", "gold": "gold 有误",
                "judge": "评测 LLM 判错", "unclear": "无法判断或需拆分"}
LABEL_OWNERS = {label: owner for owner, label in OWNER_LABELS.items()}
OBSERVATION_NAMES = {"cluster": "归因-失败簇", "unit": "归因-单个失败单元"}
# One wording for the AI prompt and the human question.
OWNER_DEFINITIONS = {
    "agent": "Agent 的输出违背商品原文或已定规则（Research 抽取、Market 聚合的行为本身有错）。",
    "rubric": "规则、检查器、阈值或评分政策不合适：检查器误判、overlay 关键词、AG-4 是否接受拆分等。",
    "gold": "gold 项的定义、取值、kind/type 或证据有错或有遗漏。",
    "judge": "评测器中的 LLM 判错：EvalAgent 的维度匹配（会造成缺失/合并/拆分）或字段打分。",
    "unclear": "证据不足以判断，或簇内单元的负责方不一致（在评论里用“拆:”列出单元）。",
}
OWNER_DESCRIPTION = "这类失败该由谁负责：" + "；".join(f"{OWNER_LABELS[k]} = {v}" for k, v in OWNER_DEFINITIONS.items())
QUESTION = "这类失败该由谁负责？只看本任务给出的证据，在右侧“失败负责方”选一项，并点旁边的评论图标写理由。不要填 human_correct。"
MATCH_RESULTS = {"missing": "缺失：gold 有这个维度，Agent 输出里没有配上的", "extra": "多出：Agent 输出有，gold 里没有对应维度"}
LABELS = {"criterion": "criteria（质量标准，有优劣方向）", "attribute": "attributes（分类属性，无优劣）",
          "boolean": "boolean（有/无）", "numeric": "numeric（数值）", "categorical": "categorical（枚举）"}
SEVERITY = {"veto": "一票否决：任一单元失败，整次交付即被阻断（block）",
            "gate": "门槛：通过率低于阈值 {threshold} 时阻断",
            "monitor": "仅监控：只报告通过率，不影响门禁结论"}
GLOSSARY = {
    "Agent 输出（actual）": "Market Agent 读完 9 款手柄的商品页后，整理出的选购维度清单。",
    "gold": "人工整理的参考维度清单，含每个维度的定义和商品原文证据。目前是草稿，尚未整体批准，本身也可能有错。",
    "匹配": "评测器里的 LLM（EvalAgent）先把 gold 维度和输出维度配对。missing = gold 有、输出里没配上；"
            "extra = 输出有、gold 里没有；合并 = 多个 gold 对 1 个输出；拆分 = 1 个 gold 对多个输出。",
    "kind": "维度属于 criteria（有优劣方向，可用来比较好坏，如续航越长越好）还是 attributes（无优劣，用于分类或筛选，如连接方式）。",
    "type": "维度取值的类型：boolean（有/无）、numeric（数值+单位）、categorical（枚举）。",
    "字段": "kind/type 决定维度需要哪些字段：units（单位）、direction（优劣方向）、values（枚举值）、formula、value_domain。",
    "根失败 / 连带失败": "程序把由同一处错误推出来的失败挂到上游那一条（根失败）下面，例如 kind 判错后 direction 字段必然多出或缺少；连带失败不单独计数。",
    "run": "一次完整的评测实验，用 run 目录名（时间戳）区分。",
}
MATCHING_RUBRICS = {"AG-1", "AG-2", "AG-4"}
FIELD_RUBRICS = {"AG-3", "AG-5"}

OWNER_PROMPT = """你在为商品维度评测的失败簇做初步归因：这类失败该由谁负责。负责方只能从下列取值中选：
""" + "\n".join(f"- {k}：{v}" for k, v in OWNER_DEFINITIONS.items()) + """

判断依据：商品原文证据、gold 项、实际输出、检查器理由，以及人工已经做出的裁决（以裁决为准）。
gold 尚未整体批准，不要默认 gold 正确；也不要默认 Agent 正确。
如果簇里个别单元的负责方明显不同，把它们列进 exceptions（用输入中的单元编号），其余单元按簇整体归因。
没有充分证据时选 unclear。reason 用一两句中文写出关键证据。"""


class UnitOwner(BaseModel):
    unit: str
    ai_owner: Literal[OWNERS]
    reason: str


class ClusterOwner(BaseModel):
    ai_owner: Literal[OWNERS]
    reason: str
    exceptions: list[UnitOwner]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit_key(ref: dict) -> str:
    return f"{ref['run_dir']}|{ref['rubric_id']}|{ref['unit']}"


def trace_url(run: dict, experiment_url: str | None) -> str | None:
    if not experiment_url or not run.get("trace_id"):
        return None
    return f"{experiment_url.split('/datasets/')[0]}/traces/{run['trace_id']}"


def label(value: str) -> str:
    return LABELS.get(value, value)


def rule_explanation(rubric_id: str, rubrics: RubricSet, overlay: GoldOverlay) -> dict:
    """The rule as a reviewer needs it: what is checked, how, and what a failure blocks."""
    r = rubrics.by_id[rubric_id]
    rule = {"规则": f"{r.id}：{r.question}", "通过条件": r.pass_, "失败条件": r.fail,
            "判定方式": "代码确定性检查（不经过 LLM）" if r.judge == "code"
                        else "LLM 裁判打分（shadow：只报告，不影响门禁）",
            "级别": SEVERITY[r.severity].format(threshold=r.threshold),
            "是否依赖 gold": "是：gold 未批准时，失败只记为待复核（needs_review），不直接阻断" if r.depends_on_gold
                             else "否：不参照 gold，只看输出本身",
            "数据来源": r.data_source}
    if rubric_id == "AG-6":
        rule["排除关键词"] = overlay.excluded_keywords
    if rubric_id in ("AG-1", "AG-4"):
        rule["核心维度"] = overlay.core_ids
    return rule


def explain_unit(unit: dict) -> str:
    """Why this unit failed, in plain words, from the checker's reason."""
    name = f"「{unit.get('name') or unit['object']}」（{unit['object']}）"
    field, reason = unit["field"], unit["reason"]
    if m := re.search(r"gold is (\w+), actual is \[(.*)\]", reason):
        actual = "、".join(label(a.strip(" '")) for a in m[2].split(","))
        return f"gold 把{name}的 {field} 定为 {label(m[1])}，Agent 输出为 {actual}。两者不一致，判为失败。"
    if m := re.search(r"field exists on (actual|gold) only", reason):
        side = "Agent 输出" if m[1] == "actual" else "gold"
        return (f"{name}的 {field} 字段只在{side}里有。是否需要这个字段由 kind/type 决定"
                f"（例如只有 criteria 才有 direction），所以这通常是 kind/type 不一致带出来的。")
    if m := re.search(r"(\d+) gold to (\d+) actual", reason):
        if int(m[1]) > 1:
            return (f"匹配时，{m[1]} 个 gold 维度（含核心维度{name}）对应到同 1 个输出维度，"
                    f"即 Agent 把它们合并成了一个。AG-4 要求核心维度一对一。")
        return f"匹配时，核心维度{name}对应到 {m[2]} 个输出维度，即 Agent 把它拆成了多个。AG-4 要求核心维度一对一。"
    if reason.startswith("core dimension is missing"):
        return f"核心维度{name}在 Agent 输出中找不到对应维度（匹配结果为 missing）。核心维度必须全部出现。"
    if reason == "dimension is missing from Market":
        return f"非核心维度{name}在 Agent 输出中找不到对应维度（匹配结果为 missing）。AG-2 只做监控，不影响门禁。"
    if m := re.search(r"excluded keywords: \[(.*)\]", reason):
        return f"Agent 输出的维度{name}的名称或别名含排除关键词 {m[1]}，被视为与商品本身无关的维度。"
    if "mixed group" in reason:
        return f"{name}匹配到的多个输出维度 kind/type 不一致，代码无法判断对错，记为未决（unresolved）。"
    if "must be a substring of the product text" in reason:
        return f"Research 对{name}记录的引文 source_text 在该商品冻结原文里找不到（去掉空白后逐字比对，不做其他归一化）。"
    if m := re.search(r"\[(.*)\] not found in source_text", reason):
        return f"Research 对{name}抽取的 {m[1]} 没有逐字出现在它自己的引文 source_text 中。"
    if reason == DERIVED_FROM_TYPE:
        return f"{name}的 {field} 分不是 LLM 判的，而是根据 kind/type 或字段是否存在推导出来的。"
    if unit["rubric"] == "AG-5":
        return f"LLM 字段裁判认为{name}的 {field} 与 gold 语义不一致：{unit.get('note') or reason}"
    return f"{name}：{reason}"


def display_example(example: dict, records: dict) -> dict:
    """A cluster example with Chinese keys; dimension contents keep their schema field names."""
    shown = {"单元": f"{Path(example['unit']['run_dir']).name} {example['unit']['rubric_id']} {example['unit']['unit']}",
             "失败原因": explain_unit(records[unit_key(example["unit"])])}
    if "evidence" in example:
        shown["Research 抽取的证据"] = example["evidence"]
    if "gold" in example:
        shown["gold 维度"] = example["gold"]
    if example.get("gold_evidence"):
        shown["gold 原文证据"] = [{"商品": e["item_id"], "引文": e["quote"]} for e in example["gold_evidence"]]
    if "actual" in example:
        shown["Agent 输出维度"] = example["actual"]
    matching = example.get("matching")
    if matching and "result" in matching:
        shown["匹配结果"] = {"结论": MATCH_RESULTS[matching["result"]], "匹配理由": matching["reason"]}
    elif matching:
        shown["匹配结果"] = {"同组 gold 维度": matching["gold"], "同组 Agent 输出维度": matching["actual"],
                         "匹配理由": matching["reason"]}
    return shown


def propose_owners(result: dict, model, decisions_path=ROOT / "eval/cases/gamepad/human-review-decisions.json") -> dict:
    """AI owner per cluster, with exception units. Stored locally; never shown before the human label."""
    records = {unit_key(u["ref"]): u for u in result["units"]}
    decisions = read_json(decisions_path)["items"]
    case = read_json(Path(result["runs"][0]["run_dir"]) / "case.json")
    rubrics = load_rubrics()
    owners = {}
    for cluster in result["clusters"]:
        numbered = {f"u{i}": ref for i, ref in enumerate(cluster["units"])}
        payload = dict(
            name=cluster["name"], summary=cluster["summary"], signatures=cluster["merged_signatures"],
            rubrics={r: rubrics.by_id[r].question for r in cluster["rubric_ids"]},
            gold_status=case["metadata"].get("status"), examples=cluster["examples"],
            human_decisions=[d for d in decisions if d["id"] in cluster["dimension_ids"]],
            units={k: dict(rubric=ref["rubric_id"], unit=ref["unit"], explanation=explain_unit(records[unit_key(ref)]))
                   for k, ref in numbered.items()})

        def check(owner: ClusterOwner):
            if unknown := [e.unit for e in owner.exceptions if e.unit not in numbered]:
                raise ValueError(f"exceptions reference unknown units: {unknown}")
            if len(owner.exceptions) >= len(numbered):
                raise ValueError("exceptions cannot cover every unit; change the cluster owner instead")

        owner = structured(model, ClusterOwner, OWNER_PROMPT, payload, check=check)
        owners[cluster["cluster_id"]] = dict(
            ai_owner=owner.ai_owner, reason=owner.reason,
            exceptions=[dict(unit=numbered[e.unit], ai_owner=e.ai_owner, reason=e.reason)
                        for e in owner.exceptions if e.ai_owner != owner.ai_owner])
    return owners


def attribution_tasks(result: dict, owners: dict) -> tuple[list[dict], dict]:
    """One task per cluster plus one per exception unit; returns (tasks, AI owner by task ID)."""
    first_run = Path(result["runs"][0]["run_dir"])
    case = read_json(first_run / "case.json")
    rubric_hashes = read_json(first_run / "rubric.json")["rubric_hashes"]
    records = {unit_key(u["ref"]): u for u in result["units"]}
    experiment = {r["run_dir"]: r.get("experiment_url") for r in result["runs"]}
    rubrics, overlay = load_rubrics(), load_overlay(DEFAULT_OVERLAY)
    tasks, ai = [], {}

    def add(kind, cluster, units, owner):
        identity = dict(task_kind=kind, cluster_id=cluster["cluster_id"], units=sorted(units, key=unit_key),
                        input_sha256=case["metadata"]["input_sha256"],
                        gold_sha256=case["metadata"]["expected_output_sha256"],
                        rubric_hashes=rubric_hashes, attribution_version=ATTRIBUTION_VERSION)
        task_id = digest(identity)
        rubric_ids = sorted({u["rubric_id"] for u in units})
        if kind == "unit":
            objects = [(u["rubric_id"].startswith("RS-"), split_unit(u["rubric_id"], u["unit"])[0]) for u in units]
            item_ids = sorted({o for research, o in objects if research})
            dimension_ids = sorted({o for research, o in objects if not research})
        else:
            item_ids, dimension_ids = cluster["item_ids"], cluster["dimension_ids"]
        runs = [r for r in cluster["runs"] if any(u["run_dir"] == r["run_dir"] for u in units)]
        tasks.append(dict(
            id=task_id, task_kind=kind, cluster_id=cluster["cluster_id"], units=units, dimension_ids=dimension_ids,
            # Langfuse shows keys in this order: what to decide, why it failed, then evidence and reference.
            input={
                "问题": QUESTION,
                "这一簇": {"名称": cluster["name"], "摘要": cluster["summary"], "簇编号": cluster["cluster_id"],
                           "失败单元数": len(units),
                           "连带失败数（未单列）": len(cluster["derived_units"]) if kind == "cluster" else 0},
                "规则说明": [rule_explanation(r, rubrics, overlay) for r in rubric_ids],
                "为什么判为失败": [{"单元": f"{Path(u['run_dir']).name} {u['rubric_id']} {u['unit']}",
                                    "说明": explain_unit(records[unit_key(u)]),
                                    "检查器原始理由": records[unit_key(u)]["reason"]} for u in units],
                "代表样例": [display_example(e, records)
                             for e in [e for e in cluster["examples"] if e["unit"] in units] or cluster["examples"][:1]],
                "负责方定义": {OWNER_LABELS[k]: v for k, v in OWNER_DEFINITIONS.items()},
                "术语": GLOSSARY,
                "链接": {"trace": [trace_url(r, experiment[r["run_dir"]]) for r in runs],
                         "Research 观测名（在 trace 中查找）": [f"research:{i}" for i in item_ids]},
            },
            observation_name=OBSERVATION_NAMES[kind],
            output={"涉及规则": rubric_ids, "涉及商品": item_ids, "涉及维度": dimension_ids,
                    "失败单元": [{"run": Path(u["run_dir"]).name, "规则": u["rubric_id"], "单元": u["unit"]} for u in units]},
            # Never put the AI owner here: metadata is visible to the reviewer.
            metadata=dict(attribution_task_id=task_id, task_kind=kind, cluster_id=cluster["cluster_id"],
                          rubric_ids=rubric_ids, item_ids=item_ids, dimension_ids=dimension_ids,
                          run_dirs=[r["run_dir"] for r in runs], trace_ids=[r["trace_id"] for r in runs],
                          input_sha256=identity["input_sha256"], gold_sha256=identity["gold_sha256"],
                          attribution_version=ATTRIBUTION_VERSION)))
        ai[task_id] = dict(cluster_id=cluster["cluster_id"], task_kind=kind, **owner)

    for cluster in result["clusters"]:
        owner = owners[cluster["cluster_id"]]
        excepted = {unit_key(e["unit"]) for e in owner["exceptions"]}
        rest = [u for u in cluster["units"] if unit_key(u) not in excepted]
        add("cluster", cluster, rest, dict(ai_owner=owner["ai_owner"], reason=owner["reason"]))
        for e in owner["exceptions"]:
            add("unit", cluster, [e["unit"]], dict(ai_owner=e["ai_owner"], reason=e["reason"]))
    return tasks, ai


def ensure_attribution_queue():
    config = ensure_score_config(SCORE_NAME, "CATEGORICAL", OWNER_DESCRIPTION,
                                 categories=[OWNER_LABELS[o] for o in OWNERS])
    # The public API cannot edit a queue's score configs; add failure_owner to the shared queue in the UI once.
    [queue] = [q for q in all_items("annotation-queues", "list") if q["name"] == QUEUE_NAME]
    if config["id"] not in queue["scoreConfigIds"]:
        raise ValueError(f"add the score config {SCORE_NAME!r} to the annotation queue {QUEUE_NAME!r} in the Langfuse UI "
                         "(Annotation Queues -> edit queue -> score configs), then publish again")
    return queue, config


def attach_ai_scores(reviews: list[dict], ai: dict, client) -> None:
    """Publish the AI owner on the reviewed observation only after the human label exists."""
    config = ensure_score_config(AI_SCORE_NAME, "CATEGORICAL", "AI 对失败负责方的初判，人工标注完成后才附上，用于对比一致率。",
                                 categories=[OWNER_LABELS[o] for o in OWNERS])
    for review in reviews:
        owner = ai[review["task_id"]]
        client.create_score(name=AI_SCORE_NAME, value=OWNER_LABELS[owner["ai_owner"]], data_type="CATEGORICAL",
                            config_id=config["id"], trace_id=review["trace_id"], observation_id=review["observation_id"],
                            score_id=digest({"task": review["task_id"], "name": AI_SCORE_NAME}), comment=owner["reason"],
                            metadata={"attribution_task_id": review["task_id"], "cluster_id": owner["cluster_id"]})
    client.flush()


def owner_codes(reviews: list[dict]) -> list[dict]:
    """Human labels arrive as Chinese category labels; files and code use the English owner codes."""
    return [{**r, "human_score": LABEL_OWNERS[r["human_score"]]} for r in reviews]


def summarize(reviews: list[dict], ai: dict) -> dict:
    rows = [dict(task_id=r["task_id"], task_kind=r["task"]["task_kind"], cluster_id=r["task"]["cluster_id"],
                 human_owner=r["human_score"], comment=r["comment"], ai_owner=ai[r["task_id"]]["ai_owner"],
                 ai_reason=ai[r["task_id"]]["reason"], agree=r["human_score"] == ai[r["task_id"]]["ai_owner"],
                 reviewer=r["reviewer"]) for r in reviews]
    confusion = Counter((row["human_owner"], row["ai_owner"]) for row in rows)
    return dict(count=len(rows), agreement=sum(r["agree"] for r in rows) / len(rows) if rows else None,
                by_human_owner={o: dict(count=sum(r["human_owner"] == o for r in rows),
                                        agree=sum(r["agree"] for r in rows if r["human_owner"] == o)) for o in OWNERS},
                confusion=[dict(human=h, ai=a, count=n) for (h, a), n in sorted(confusion.items())], rows=rows)


def followup_tasks(reviews: list[dict]) -> list[dict]:
    """Scoring-queue drafts for clusters a human attributed to gold or judge."""
    tasks = {}
    for review in reviews:
        task, owner = review["task"], review["human_score"]
        if owner not in ("gold", "judge"):
            continue
        for run_dir in sorted({u["run_dir"] for u in task["units"]}):
            units = [u for u in task["units"] if u["run_dir"] == run_dir]
            case = read_json(Path(run_dir) / "case.json")
            if owner == "gold":
                candidates = [t for t in review_tasks(case) if t["task_kind"] == "gold_item" and t["ref"] in task["dimension_ids"]]
            else:
                report = EvalReport.model_validate_json((Path(run_dir) / "report.json").read_text(encoding="utf-8"))
                gold_id = {ref: entry["item"]["id"] for ref, entry in report.items.items()}
                fields = {split_unit(u["rubric_id"], u["unit"]) for u in units if u["rubric_id"] in FIELD_RUBRICS}
                matching = any(u["rubric_id"] in MATCHING_RUBRICS for u in units)
                candidates = [t for t in review_tasks(case, report)
                              if (t["task_kind"] == "schema_field" and (gold_id[t["ref"]], t["field"]) in fields)
                              or (t["task_kind"] == "matching" and matching)]
            for candidate in candidates:
                candidate["metadata"]["from_attribution_task"] = review["task_id"]
                tasks.setdefault(candidate["id"], candidate)
    return list(tasks.values())
