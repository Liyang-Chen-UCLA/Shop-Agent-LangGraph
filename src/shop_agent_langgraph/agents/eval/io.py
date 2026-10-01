"""Load annotated gold/Market exports and save reviewable offline reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...domain.criteria import CriteriaAttributeSet
from .schemas import EvalReport


ITEM_FIELDS = {"id", "name", "description", "aliases", "type", "units", "formula",
               "direction", "values", "value_domain"}


def load_gold(path: str | Path) -> tuple[CriteriaAttributeSet, dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    collection = CriteriaAttributeSet.model_validate({
        "source_item_ids": payload["source_item_ids"],
        **{kind: [{key: value for key, value in item.items() if key in ITEM_FIELDS}
                  for item in payload[kind]] for kind in ("criteria", "attributes")},
    })
    metadata = {**payload.get("annotation_metadata", {}),
                "gold_path": str(path.resolve()), "gold_sha256": hashlib.sha256(raw).hexdigest()}
    return collection, metadata


def load_actual(path: str | Path) -> CriteriaAttributeSet:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "result" in payload:  # saved Market cache envelope
        payload = payload["result"]
    if "status" in payload and payload["status"] != "completed":
        raise ValueError("actual Market delivery must be completed")
    return CriteriaAttributeSet.model_validate({
        "source_item_ids": payload.get("source_item_ids", payload.get("item_ids")),
        "criteria": payload["criteria"], "attributes": payload["attributes"],
    })


def render_report(report: EvalReport) -> str:
    def label(ref):
        entry = report.items[ref]
        return f"{entry['item']['name']} ({entry['kind']}:{entry['item']['id']})"

    def cell(text):
        return str(text).replace("|", "\\|").replace("\n", " ")

    m = report.metrics
    lines = ["# Market 与 gold 差异评估", "",
        f"固定商品范围：{', '.join(report.source_item_ids)}", "",
        f"match：{m.match_group_count} 组，覆盖 {m.matched_gold_count}/{m.gold_count} 个 gold 条目、"
        f"{m.matched_actual_count}/{m.actual_count} 个 Market 条目。",
        f"missing：{m.missing_count}；extra：{m.extra_count}。",
        f"归类错误：{m.classification_error_count} 个 matched gold 条目；"
        f"合并/拆分粒度错误：{m.granularity_error_count} 个 matched gold 条目。", "",
        "单项得分为 0/1；字段汇总为适用项的平均分，无适用项为 null。aliases 单独报告，不计入主字段汇总。",
        "extra 仅表示 gold 中没有对应维度，不代表错误或幻觉。", ""]
    if report.gold_metadata:
        lines += [f"gold 状态：{report.gold_metadata.get('status', 'unspecified')}；"
                  f"SHA256：{report.gold_metadata.get('gold_sha256', 'unspecified')}", ""]
    lines += ["## 覆盖统计", "", "| 类别 | gold | Market | matched gold | matched Market | missing | extra |",
              "|---|---|---|---|---|---|---|"]
    for kind, c in m.by_kind.items():
        lines.append(f"| {kind} | {c.gold_count} | {c.actual_count} | {c.matched_gold_count} | {c.matched_actual_count} | {c.missing_count} | {c.extra_count} |")
    lines += ["", "## 字段平均分", "", "| 字段 | 适用项平均分 |", "|---|---|"]
    for field, score in m.field_scores.items():
        lines.append(f"| {field} | {'null' if score is None else f'{score:.3f}'} |")
    for name, entries in (("missing", report.missing), ("extra", report.extra)):
        lines += ["", f"## {name}", ""]
        lines.extend(f"- {label(e.ref)}：{e.reason}" for e in entries)
        if not entries:
            lines.append("无。")
    reviews = {r.gold_ref: r for r in report.reviews}
    lines += ["", "## match 与字段审核", ""]
    for i, group in enumerate(report.match, 1):
        lines += [f"### match {i}", "", "gold：" + "; ".join(label(r) for r in group.gold_ids),
                  "Market：" + "; ".join(label(r) for r in group.actual_ids), group.reason, ""]
        for ref in group.gold_ids:
            lines += [f"**{label(ref)}**", "", "| 字段 | 0/1 | 原因 |", "|---|---|---|"]
            lines.extend(f"| {s.field} | {s.score} | {cell(s.reason)} |" for s in reviews[ref].fields)
            lines.append("")
    return "\n".join(lines) + "\n"


def save_report(report: EvalReport, directory: str | Path) -> tuple[Path, Path]:
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = directory / "report.json", directory / "report.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_report(report), encoding="utf-8")
    return json_path, markdown_path
