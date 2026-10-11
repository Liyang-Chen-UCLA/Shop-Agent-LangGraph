"""Cluster rubric failures of saved runs into root causes for human review.

1. normalize (code): failing units -> (run, rubric, object, field, signature); research
   units get text features.
2. cascade (code): failures implied by an upstream failure (a merge, a kind/type error)
   are attached to that root instead of being clustered.
3. group (code): roots with the same rubric + signature form one initial group, across runs.
4. merge/split (LLM): roots that one fix would remove together form a cluster with a
   hypothesis; coverage is checked by code.

Clustering does not attribute blame (agent, rubric, gold, judge); `attribution` does.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from .cases import write_json
from .rubric import load_rubrics, rubric_file_hashes
from .rubric_run import load_runs

CLUSTER_VERSION = "v3"  # v3: MVP rubrics (AG-4 merged into AG-1; AG-2/AG-5/RS-4/RS-5 removed)

CLUSTER_PROMPT = """你在为 Agent 评测的失败做根因聚类。输入是程序按"rubric + 错误签名"分好的初始组，每组带全部根失败单元和代表样例。
连带失败已由程序挂到根失败下，不需要再处理。

判断标准只有一条：如果只改一处，这些失败会不会一起消失？
- 会一起消失：合并，即使跨 rubric（例如同一次维度合并同时造成核心维度未一对一匹配和 kind/type 不一致）。
- 改法不同：拆开，即使签名相同。
- 没有充分理由时保持程序分组不变。

只做聚类，不判断是 Agent、gold、评分政策还是检查器的问题。
输出要求：
- 每个根失败单元都要归入一个聚类；整组归入用 groups，单独的单元用 units。
- 拆分：把要拿出来的单元列在另一个聚类的 units 里即可，它会自动从原来整组所在的聚类中移除。
- 同一单元或同一组被多个聚类认领时，归第一个认领的聚类（单独列出优先于整组）。
- hypothesis 写成"如果改 X，这些失败会一起消失"，X 写输出或判定中具体的行为。它会展示给负责归因的人工，
  不能暗示谁该负责：不写"按 gold 修正""补齐 gold"，也不写"Agent 错了""规则太严"。
- evidence 列 1-3 个最能支撑假设的单元 ID，必须属于该聚类。
- name 用简短中文。"""


class ProposedCluster(BaseModel):
    name: str
    groups: list[str] = Field(default_factory=list, description="Initial group IDs taken whole")
    units: list[str] = Field(default_factory=list, description="Individual root unit IDs")
    hypothesis: str
    evidence: list[str]


class ClusterProposal(BaseModel):
    clusters: list[ProposedCluster]


# --- 1. normalize --------------------------------------------------------------------

TEMPLATES = [
    (re.compile(r"gold is (\w+), actual is \[(.*)\]"), lambda m, f: f"{f}: {m[1]}→{m[2].replace(chr(39), '')}"),
    (re.compile(r"field exists on (actual|gold) only"), lambda m, f: f"{f}: 仅 {m[1]} 有"),
    (re.compile(r"(\d+) gold to (\d+) actual"),
     lambda m, f: "多个 gold 合并为 1 个输出" if int(m[1]) > 1 else "1 个 gold 拆成多个输出"),
    (re.compile(r"excluded keywords"), lambda m, f: "输出含排除类关键词"),
    (re.compile(r"\[(.*)\] not found in source_text"), lambda m, f: f"{m[1].replace(chr(39), '')} 不在引文中"),
    (re.compile(r"core dimension is missing"), lambda m, f: "核心维度缺失"),
]


def split_unit(rubric: str, unit: str) -> tuple[str, str]:
    """`delivery/<dim>.<field>` -> (dim, field); `<item>/e3` -> (item, e3)."""
    head, _, tail = unit.partition("/")
    if head == "delivery":
        obj, _, field = tail.partition(".")
        return obj, field
    return head, tail


def template(reason: str, field: str) -> str | None:
    for pattern, render in TEMPLATES:
        if m := pattern.search(reason):
            return render(m, field)
    return None


def research_feature(rubric: str, evidence: dict) -> str:
    """Name the text shape behind a research failure so different mechanisms group apart."""
    source = evidence.get("source_text") or ""
    if "$" in source or "\\" in source:
        return "引文含 LaTeX/公式写法"
    if rubric == "RS-1":
        if "..." in source or "…" in source:
            return "引文用省略号拼接"
        if "|" in source:
            return "引文跨表格单元"
        return "引文改写或跨段拼接"
    value = str(evidence.get("value") or "")
    parts = [p for p in re.split(r"[/、,，|；;]", value) if p.strip()]
    squashed = re.sub(r"\s+", "", source)
    if len(parts) > 1 and all(re.sub(r"\s+", "", p) in squashed for p in parts):
        return "value 由多处原文拼接"
    return "value 为改写或概括"


def load_units(run_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    runs = load_runs(run_dirs)
    units, contexts = [], []
    for index, (run_dir, run) in enumerate(zip(run_dirs, runs)):
        run_dir = Path(run_dir)
        report_path, market_path = run_dir / "report.json", run_dir / "market.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        market = json.loads(market_path.read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        research = {a["item_id"]: a["evidence"] for a in market.get("audit", []) if a.get("stage") == "fixed_research"}
        contexts.append(dict(run_dir=run_dir.as_posix(), trace_id=next(iter(manifest.get("trace_ids") or []), None),
                             experiment_url=manifest.get("experiment_url"), case=run["case"], report=report,
                             research=research))
        # Gold and delivery dimensions can share an ID; AG-6/AG-7 judge the delivery, the rest judge gold.
        names = {side: {e["item"]["id"]: e["item"]["name"] for r, e in (report or {"items": {}})["items"].items()
                        if r.startswith(side)} for side in ("gold:", "actual:")}
        for rubric, results in run["trial"].items():
            for r in results:
                if r.status not in ("fail", "unresolved"):
                    continue
                obj, field = split_unit(rubric, r.unit)
                signature = template(r.reason, field)
                if rubric.startswith("RS-") and obj in research and field.startswith("e"):
                    signature = f"{signature or r.reason} / {research_feature(rubric, research[obj][int(field[1:])])}"
                if rubric.startswith("RS-") and obj in research and field.startswith("e"):
                    name = research[obj][int(field[1:])].get("name")
                else:
                    name = names["actual:" if rubric in ("AG-6", "AG-7") else "gold:"].get(obj)
                units.append(dict(id=f"r{index}|{rubric}|{r.unit}", run=index, rubric=rubric, unit=r.unit, name=name,
                                  ref=dict(run_dir=run_dir.as_posix(), rubric_id=rubric, unit=r.unit),
                                  object=obj, field=field, status=r.status, reason=r.reason,
                                  signature=(signature or r.reason) if r.status == "fail" else f"unresolved: {signature or r.reason}",
                                  root=None))
    return units, contexts


# --- 2. cascade ----------------------------------------------------------------------

def gold_groups(report: dict | None) -> dict[str, set[str]]:
    """Gold dimension ID -> all gold dimension IDs in its matching group."""
    if not report:
        return {}
    ids = {ref: entry["item"]["id"] for ref, entry in report["items"].items()}
    groups = {}
    for group in report["match"]:
        members = {ids[r] for r in group["gold_ids"]}
        for member in members:
            groups[member] = members
    return groups


def attach_cascades(units: list[dict], contexts: list[dict]) -> None:
    """AG-3 failures on a dimension merged into one output hang off that AG-1 merge; field-presence
    failures hang off the same dimension's kind/type failure."""
    for index, context in enumerate(contexts):
        run = [u for u in units if u["run"] == index]
        kind_type = {}
        for u in run:
            if u["rubric"] == "AG-3" and u["field"] in ("kind", "type"):
                kind_type.setdefault(u["object"], u)
        merges = {u["object"]: u for u in run if u["rubric"] == "AG-1" and u["signature"].startswith("多个 gold")}
        groups = gold_groups(context["report"])
        merged_into = {member: root for obj, root in merges.items() for member in groups.get(obj, {obj})}
        for u in run:
            if u["rubric"] != "AG-3":
                continue
            if u["object"] in merged_into:
                u["root"] = merged_into[u["object"]]["id"]
            elif u["field"] not in ("kind", "type") and u["object"] in kind_type:
                u["root"] = kind_type[u["object"]]["id"]
    by_id = {u["id"]: u for u in units}
    for u in units:  # point every derived unit at its ultimate root
        while u["root"] and by_id[u["root"]]["root"]:
            u["root"] = by_id[u["root"]]["root"]


# --- 3. group ------------------------------------------------------------------------

def initial_groups(units: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for u in units:
        if u["root"] is None:
            buckets[(u["rubric"], u["signature"])].append(u)
    ordered = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    # The ID depends only on the signature rules, so the same kind of failure keeps its ID across runs.
    return [dict(id=f"{rubric}:{signature}", rubric=rubric, signature=signature, units=[u["id"] for u in members])
            for (rubric, signature), members in ordered]


# --- 4. merge / split ----------------------------------------------------------------

def sample_context(unit: dict, contexts: list[dict]) -> dict:
    context = contexts[unit["run"]]
    if unit["rubric"].startswith("RS-") and unit["field"].startswith("e"):
        return {"evidence": context["research"][unit["object"]][int(unit["field"][1:])]}
    report = context["report"]
    if not report:
        return {}
    # Gold and delivery dimensions can share an ID; AG-6/AG-7 judge the delivery, the rest judge gold.
    side = "actual:" if unit["rubric"] in ("AG-6", "AG-7") else "gold:"
    ref = next((r for r, e in report["items"].items() if r.startswith(side) and e["item"]["id"] == unit["object"]), None)
    if ref is None:
        return {}
    detail = {side.rstrip(":"): report["items"][ref]}
    if side == "gold:":
        annotation = context["case"]["gold_annotation"]
        evidence = next((i.get("evidence", []) for k in ("criteria", "attributes") for i in annotation[k]
                         if i["id"] == unit["object"]), [])
        detail["gold_evidence"] = [dict(item_id=e["item_id"], quote=e["quote"]) for e in evidence[:6]]
    for group in report["match"]:
        if ref in group["gold_ids"] + group["actual_ids"]:
            detail["matching"] = {"gold": [report["items"][r]["item"]["id"] for r in group["gold_ids"]],
                                  "actual": {r: report["items"][r] for r in group["actual_ids"]},
                                  "reason": group["reason"]}
    for key in ("missing", "extra"):
        if reason := next((m["reason"] for m in report[key] if m["ref"] == ref), None):
            detail["matching"] = {"result": key, "reason": reason}
    return detail


def cluster_payload(groups: list[dict], units: list[dict], contexts: list[dict]) -> dict:
    by_id = {u["id"]: u for u in units}
    derived = defaultdict(int)
    for u in units:
        if u["root"]:
            derived[u["root"]] += 1
    rubrics = load_rubrics()
    return {"runs": [c["run_dir"] for c in contexts], "groups": [dict(
        id=g["id"], rubric=g["rubric"], question=rubrics.by_id[g["rubric"]].question, signature=g["signature"],
        units=[dict(id=i, object=by_id[i]["object"], field=by_id[i]["field"], reason=by_id[i]["reason"][:200],
                    derived=derived[i]) for i in g["units"]],
        samples=[dict(id=i, **sample_context(by_id[i], contexts)) for i in g["units"][:2]])
        for g in groups]}


def expand(proposal: ClusterProposal, groups: list[dict]) -> list[list[str]]:
    """Root units of each proposed cluster.

    Each unit goes to the first cluster that claims it, individual listings before whole
    groups, so a unit singled out of its group leaves it and a doubly claimed one is not lost.
    """
    members = {g["id"]: g["units"] for g in groups}
    owner: dict[str, int] = {}
    for index, c in enumerate(proposal.clusters):
        for u in c.units:
            owner.setdefault(u, index)
    for index, c in enumerate(proposal.clusters):
        for g in c.groups:
            for u in members.get(g, []):
                owner.setdefault(u, index)
    return [[u for u, i in owner.items() if i == index] for index in range(len(proposal.clusters))]


def check_proposal(proposal: ClusterProposal, groups: list[dict]) -> None:
    known_groups = {g["id"] for g in groups}
    roots = {i for g in groups for i in g["units"]}
    for c in proposal.clusters:
        if unknown := [g for g in c.groups if g not in known_groups] + [u for u in c.units if u not in roots]:
            raise ValueError(f"cluster {c.name!r} references unknown IDs: {unknown}")
    if omitted := sorted(roots - {u for units in expand(proposal, groups) for u in units}):
        raise ValueError(f"every root unit must be in a cluster; omitted={omitted}")


def structured(model, schema, system: str, payload, check=lambda result: None):
    bound = model.with_structured_output(schema, method="function_calling")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]
    for attempt in range(2):
        raw = bound.invoke(messages)
        result = raw if isinstance(raw, schema) else schema.model_validate(raw)
        try:
            check(result)
            return result
        except ValueError as error:
            if attempt:
                raise
            messages = messages[:2] + [{"role": "user", "content": f"上一次输出：{result.model_dump_json()}\n"
                                        f"被拒绝：{error}。请针对原始输入返回完整的修正结果。"}]


def build_clusters(proposal: ClusterProposal, groups: list[dict], units: list[dict], contexts: list[dict]) -> list[dict]:
    members = {g["id"]: g for g in groups}
    group_of = {i: g["id"] for g in groups for i in g["units"]}
    by_id = {u["id"]: u for u in units}
    derived = defaultdict(list)
    for u in units:
        if u["root"]:
            derived[u["root"]].append(u["id"])
    clusters, taken = [], defaultdict(int)
    for c, roots in zip(proposal.clusters, expand(proposal, groups)):
        if not roots:
            continue  # every unit it claimed went to an earlier cluster
        sizes = defaultdict(int)
        for u in roots:
            sizes[group_of[u]] += 1
        merged_from = sorted(sizes, key=lambda g: (-sizes[g], g))
        anchor = merged_from[0]
        taken[anchor] += 1
        unchanged = len(merged_from) == 1 and set(roots) == set(members[anchor]["units"])
        research = [by_id[u] for u in roots if by_id[u]["rubric"].startswith("RS-")]
        clusters.append(dict(
            cluster_id=anchor if taken[anchor] == 1 else f"{anchor}#{taken[anchor]}",
            name=c.name, summary=c.hypothesis, changed_by_llm=not unchanged, merged_signatures=merged_from,
            rubric_ids=sorted({by_id[u]["rubric"] for u in roots}),
            runs=[dict(run_dir=contexts[i]["run_dir"], trace_id=contexts[i]["trace_id"])
                  for i in sorted({by_id[u]["run"] for u in roots})],
            units=[by_id[u]["ref"] for u in roots],
            derived_units=[by_id[d]["ref"] for u in roots for d in derived[u]],
            item_ids=sorted({u["object"] for u in research}),
            dimension_ids=sorted({by_id[u]["object"] for u in roots if not by_id[u]["rubric"].startswith("RS-")}),
            examples=[dict(unit=by_id[e]["ref"], reason=by_id[e]["reason"],
                           **sample_context(by_id[e], contexts))
                      for e in [e for e in c.evidence if e in roots] or roots[:1]]))  # examples only illustrate
    return clusters


def cluster_failures(run_dirs: list[str | Path], model) -> dict:
    run_dirs = [Path(d) for d in run_dirs]
    units, contexts = load_units(run_dirs)
    attach_cascades(units, contexts)
    groups = initial_groups(units)
    if groups:
        proposal = structured(model, ClusterProposal, CLUSTER_PROMPT, cluster_payload(groups, units, contexts),
                              check=lambda p: check_proposal(p, groups))
    else:
        proposal = ClusterProposal(clusters=[])
    return {"version": CLUSTER_VERSION, "rubric_hashes": rubric_file_hashes(), "runs": [dict(run_dir=c["run_dir"], trace_id=c["trace_id"],
                                                      experiment_url=c["experiment_url"]) for c in contexts],
            "groups": groups, "clusters": build_clusters(proposal, groups, units, contexts), "units": units}


def render_markdown(result: dict) -> str:
    roots = sum(1 for u in result["units"] if u["root"] is None)
    runs = {r["run_dir"]: f"r{i}" for i, r in enumerate(result["runs"])}
    lines = ["# 失败聚类", "",
             f"runs：{', '.join(f'{v}={k}' for k, v in runs.items())}；失败单元 {len(result['units'])}，"
             f"根失败 {roots}，初始组 {len(result['groups'])}，聚类 {len(result['clusters'])}。",
             "标 ⚠ 的聚类由 LLM 合并或拆分了程序分组。本文件由 clusters.json 渲染。", "",
             "| 聚类 | 名称 | 根失败 | 连带 | 出现 run | 签名 |", "|---|---|---:|---:|---|---|"]
    for c in result["clusters"]:
        flag = "⚠ " if c["changed_by_llm"] else ""
        lines.append(f"| `{c['cluster_id']}` | {flag}{c['name']} | {len(c['units'])} | {len(c['derived_units'])} | "
                     f"{', '.join(runs[r['run_dir']] for r in c['runs'])} | {'<br>'.join(c['merged_signatures'])} |")
    for c in result["clusters"]:
        scope = c["item_ids"] and f"商品：{', '.join(c['item_ids'])}" or f"维度：{', '.join(c['dimension_ids'])}"
        lines += ["", f"## {'⚠ ' if c['changed_by_llm'] else ''}{c['name']}", "", f"`{c['cluster_id']}`", "",
                  f"**摘要**：{c['summary']}", "", scope, "", "代表样例："]
        for e in c["examples"]:
            u = e["unit"]
            lines.append(f"- {runs[u['run_dir']]} {u['rubric_id']} `{u['unit']}`：{e['reason']}")
    return "\n".join(lines) + "\n"


def write_clusters(result: dict, directory: str | Path) -> None:
    directory = Path(directory)
    write_json(directory / "clusters.json", result)
    (directory / "clusters.md").write_text(render_markdown(result), encoding="utf-8")
