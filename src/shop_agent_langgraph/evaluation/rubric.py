"""Explicit, validated rubrics: YAML assets, loaders and (below) the scoring gate."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RUBRIC_DIR = Path(__file__).with_name("rubrics")
MODULE_PREFIXES = {"e2e": "E2E", "intent": "INT", "route": "RT", "market_screening": "MS",
                   "research": "RS", "aggregation": "AG"}
ID_PATTERN = re.compile(r"^[A-Z0-9]+-\d+$")


class UnitResult(BaseModel):
    """Outcome of one checked unit (an evidence, a dimension, a tool call, a turn...)."""
    model_config = ConfigDict(extra="forbid")

    unit: str = Field(min_length=1)
    status: Literal["pass", "fail", "unresolved", "na"]
    reason: str = ""


class RubricField(BaseModel):
    """One schema sub-field; its definition is the single source for every judge's wording."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    auxiliary: bool = False

    @field_validator("definition")
    @classmethod
    def normalized(cls, value: str) -> str:
        return " ".join(value.split())


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    question: str = Field(min_length=1)
    pass_: str = Field(alias="pass", min_length=1)
    fail: str = Field(min_length=1)
    judge: Literal["code", "llm"]
    severity: Literal["veto", "gate", "monitor"]
    threshold: float | None = None
    maturity: Literal["shadow", "active"]
    depends_on_gold: bool
    data_source: str = Field(min_length=1)
    review_question: str | None = None
    fields: list[RubricField] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_policy(self) -> Rubric:
        if not ID_PATTERN.match(self.id):
            raise ValueError(f"invalid rubric id {self.id!r}")
        if self.judge == "llm" and self.maturity != "shadow":
            raise ValueError(f"{self.id}: llm rubrics must start as shadow")
        if self.judge == "code" and self.maturity != "active":
            raise ValueError(f"{self.id}: code rubrics must be active")
        if self.severity == "gate":
            if self.threshold is None or not 0 < self.threshold <= 1:
                raise ValueError(f"{self.id}: gate rubrics need a threshold in (0, 1]")
        elif self.threshold is not None:
            raise ValueError(f"{self.id}: only gate rubrics may define a threshold")
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.id}: duplicate field definitions")
        if self.fields and not (self.review_question and "{field}" in self.review_question):
            raise ValueError(f"{self.id}: rubrics with fields need a review_question containing {{field}}")
        return self


class RubricFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: Literal["e2e", "intent", "route", "market_screening", "research", "aggregation"]
    version: int = Field(ge=1)
    rubrics: list[Rubric] = Field(min_length=1)

    @model_validator(mode="after")
    def check_ids(self) -> RubricFile:
        prefix = MODULE_PREFIXES[self.module]
        ids = [r.id for r in self.rubrics]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.module}: duplicate rubric ids")
        if any(not r.id.startswith(prefix + "-") for r in self.rubrics):
            raise ValueError(f"{self.module}: rubric ids must start with {prefix}-")
        return self


class RubricSet:
    def __init__(self, files: dict[str, RubricFile]):
        self.files = files
        self.rubrics = [r for module in MODULE_PREFIXES if module in files for r in files[module].rubrics]
        self.by_id = {r.id: r for r in self.rubrics}
        if len(self.by_id) != len(self.rubrics):
            raise ValueError("rubric ids must be unique across modules")

    def module(self, name: str) -> list[Rubric]:
        return list(self.files[name].rubrics)

    def module_of(self, rubric_id: str) -> str:
        return next(m for m, f in self.files.items() if any(r.id == rubric_id for r in f.rubrics))


def load_rubrics(directory: str | Path | None = None) -> RubricSet:
    """Load and validate every module file; each module must have exactly its own YAML."""
    directory = Path(directory or RUBRIC_DIR)
    files: dict[str, RubricFile] = {}
    for module in MODULE_PREFIXES:
        path = directory / f"{module}.yaml"
        if not path.exists():
            raise ValueError(f"missing rubric file {path.name}")
        parsed = RubricFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        if parsed.module != module:
            raise ValueError(f"{path.name} declares module {parsed.module!r}")
        files[module] = parsed
    extra = {p.stem for p in directory.glob("*.yaml")} - set(MODULE_PREFIXES)
    if extra:
        raise ValueError(f"unknown rubric files: {sorted(extra)}")
    return RubricSet(files)


def rubric_file_hashes(directory: str | Path | None = None) -> dict[str, str]:
    directory = Path(directory or RUBRIC_DIR)
    return {f"evaluation/rubrics/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.yaml"))}


class GoldOverlay(BaseModel):
    """Per-case, gold-dependent rubric parameters kept out of the generic rubrics."""
    model_config = ConfigDict(extra="forbid")

    status: str = Field(min_length=1)
    core_ids: list[str] = Field(min_length=1)
    excluded_keywords: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique(self) -> GoldOverlay:
        if len(self.core_ids) != len(set(self.core_ids)):
            raise ValueError("core_ids must be unique")
        if any(not k.strip() for k in self.excluded_keywords):
            raise ValueError("excluded_keywords must be nonblank")
        return self

    @property
    def approved(self) -> bool:
        return self.status == "approved"


def load_overlay(path: str | Path) -> GoldOverlay:
    return GoldOverlay.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# --- Scoring gate ----------------------------------------------------------------------

Verdict = Literal["pass", "block", "needs_review", "report", "na", "not_run"]
Conclusion = Literal["pass", "needs_review", "block"]
EPSILON = 1e-9


class RubricOutcome(BaseModel):
    """A rubric's aggregate over one or more trials, plus its gate verdict."""
    model_config = ConfigDict(extra="forbid")

    id: str
    severity: str
    maturity: str
    judge: str
    threshold: float | None = None
    trials: int
    counts: dict[str, int]
    pass_rate: float | None = None
    optimistic_rate: float | None = None
    verdict: Verdict
    would_be: Verdict | None = None
    gold_downgraded: bool = False
    reason: str = ""
    failures: list[str] = Field(default_factory=list)


class GateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conclusion: Conclusion
    gold_approved: dict[str, bool]
    trial_count: int
    outcomes: list[RubricOutcome]
    blockers: list[str]
    reviews: list[str]


def _tally(units: list[UnitResult]) -> dict[str, int]:
    return {status: sum(u.status == status for u in units) for status in ("pass", "fail", "unresolved", "na")}


def _rate(counts: dict[str, int], optimistic: bool = False) -> float | None:
    denominator = counts["pass"] + counts["fail"] + counts["unresolved"]
    if not denominator:
        return None
    return (counts["pass"] + (counts["unresolved"] if optimistic else 0)) / denominator


def evaluate_rubric(rubric: Rubric, trials: list[list[UnitResult]], *, gold_approved: bool) -> RubricOutcome:
    """Aggregate trials: veto needs every trial clean; gate compares the mean per-trial pass rate."""
    per_trial = [_tally(units) for units in trials]
    total = {key: sum(c[key] for c in per_trial) for key in ("pass", "fail", "unresolved", "na")}
    base = dict(id=rubric.id, severity=rubric.severity, maturity=rubric.maturity, judge=rubric.judge,
                threshold=rubric.threshold, trials=len(trials), counts=total)
    failures = [f"{u.unit}: {u.reason}" if u.reason else u.unit for units in trials for u in units if u.status == "fail"][:20]
    if not any(total.values()):
        return RubricOutcome(**base, verdict="not_run", reason="no judged units (no judge or no data)")
    rates = [r for r in (_rate(c) for c in per_trial) if r is not None]
    if not rates:
        return RubricOutcome(**base, verdict="na", reason="no applicable units")
    optimistic = [_rate(c, True) for c in per_trial if _rate(c) is not None]
    mean, best = sum(rates) / len(rates), sum(optimistic) / len(optimistic)
    pooled = _rate(total)
    reason = ""
    if rubric.severity == "veto":
        raw: Verdict = "block" if total["fail"] else "needs_review" if total["unresolved"] else "pass"
        value = pooled
        reason = {"block": f"{total['fail']} failing unit(s); veto needs every trial clean",
                  "needs_review": f"{total['unresolved']} unresolved unit(s)", "pass": ""}[raw]
    elif rubric.severity == "gate":
        raw = "pass" if mean >= rubric.threshold - EPSILON else "needs_review" if best >= rubric.threshold - EPSILON else "block"
        value = mean
        reason = "" if raw == "pass" else f"mean pass rate {mean:.3f} < threshold {rubric.threshold}" + (
            " (unresolved units could still reach it)" if raw == "needs_review" else "")
    else:
        raw, value = "report", pooled
    outcome = dict(**base, pass_rate=value, optimistic_rate=best if rubric.severity == "gate" else None,
                   failures=failures, reason=reason)
    if rubric.maturity == "shadow" or rubric.severity == "monitor":
        return RubricOutcome(**outcome, verdict="report", would_be=None if raw == "report" else raw)
    if raw == "block" and rubric.depends_on_gold and not gold_approved:
        return RubricOutcome(**outcome | {"reason": f"{reason}; gold is not approved, so the failure needs review"},
                             verdict="needs_review", gold_downgraded=True)
    return RubricOutcome(**outcome, verdict=raw)


def evaluate_gate(rubrics: RubricSet, trials: list[dict[str, list[UnitResult]]], *,
                  gold_approved: bool | dict[str, bool] = False, modules: list[str] | None = None) -> GateReport:
    """Combine rubric outcomes: block > needs_review > pass. Shadow and monitor rubrics only report."""
    if modules is None:
        seen = {rubrics.module_of(rid) for trial in trials for rid in trial}
        modules = [m for m in MODULE_PREFIXES if m in seen]
    approvals = {m: (gold_approved if isinstance(gold_approved, bool) else gold_approved.get(m, False)) for m in modules}
    outcomes: list[RubricOutcome] = []
    for module in modules:
        for rubric in rubrics.module(module):
            outcomes.append(evaluate_rubric(rubric, [t[rubric.id] for t in trials if rubric.id in t],
                                            gold_approved=approvals[module]))
    blockers = [o.id for o in outcomes if o.verdict == "block"]
    reviews = [o.id for o in outcomes if o.verdict == "needs_review"]
    return GateReport(conclusion="block" if blockers else "needs_review" if reviews else "pass",
                      gold_approved=approvals, trial_count=len(trials), outcomes=outcomes,
                      blockers=blockers, reviews=reviews)


def render_gate_markdown(report: GateReport) -> str:
    def cell(text: str) -> str:
        return text.replace("|", r"\|").replace("\n", " ")

    lines = [f"# Rubric 门禁：{report.conclusion}", "",
             f"trial 数：{report.trial_count}；gold 已批准：{report.gold_approved}。",
             "block 来自 active 的 veto/gate；gold 未批准时依赖 gold 的失败降级为 needs_review；shadow 与 monitor 只报告。", "",
             "| rubric | 判定方式 | 级别 | 状态 | 通过率 | 阈值 | 结论 | 说明 |", "|---|---|---|---|---|---|---|---|"]
    for o in report.outcomes:
        rate = "-" if o.pass_rate is None else f"{o.pass_rate:.3f}"
        verdict = o.verdict if o.would_be is None else f"{o.verdict}（若 active：{o.would_be}）"
        lines.append(f"| {o.id} | {o.judge} | {o.severity} | {o.maturity} | {rate} | "
                     f"{'-' if o.threshold is None else o.threshold} | {verdict} | {cell(o.reason)} |")
    failing = [o for o in report.outcomes if o.failures]
    if failing:
        lines += ["", "## 失败单元（每条 rubric 最多 20 个）", ""]
        for o in failing:
            lines += [f"### {o.id}", ""] + [f"- {cell(f)}" for f in o.failures] + [""]
    return "\n".join(lines) + "\n"
