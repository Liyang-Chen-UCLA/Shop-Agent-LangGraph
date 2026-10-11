"""Explicit, validated rubrics: YAML assets, loaders and a pass-rate summary (no gate)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RUBRIC_DIR = Path(__file__).with_name("rubrics")
SCHEMA_FIELDS_PATH = Path(__file__).with_name("schema_fields.yaml")
MODULE_PREFIXES = {"research": "RS", "aggregation": "AG"}
ID_PATTERN = re.compile(r"^[A-Z0-9]+-\d+$")


class UnitResult(BaseModel):
    """Outcome of one checked unit (an evidence, a dimension...)."""
    model_config = ConfigDict(extra="forbid")

    unit: str = Field(min_length=1)
    status: Literal["pass", "fail", "unresolved", "na"]
    reason: str = ""


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    question: str = Field(min_length=1)
    pass_: str = Field(alias="pass", min_length=1)
    fail: str = Field(min_length=1)
    data_source: str = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def check_id(cls, value: str) -> str:
        if not ID_PATTERN.match(value):
            raise ValueError(f"invalid rubric id {value!r}")
        return value


class RubricFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: Literal["research", "aggregation"]
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


def load_overlay(path: str | Path) -> GoldOverlay:
    return GoldOverlay.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# --- Pass-rate summary ------------------------------------------------------------------

class RubricOutcome(BaseModel):
    """A rubric's pass rate over one run. Reported only; nothing blocks on it."""
    model_config = ConfigDict(extra="forbid")

    id: str
    counts: dict[str, int]
    pass_rate: float | None = None
    failures: list[str] = Field(default_factory=list)


def summarize(rubrics: RubricSet, trial: dict[str, list[UnitResult]]) -> list[RubricOutcome]:
    """Pass rate = pass / (pass + fail + unresolved); `na` is excluded; empty denominator -> None."""
    outcomes = []
    for rubric in rubrics.rubrics:
        if rubric.id not in trial:
            continue
        units = trial[rubric.id]
        counts = {status: sum(u.status == status for u in units) for status in ("pass", "fail", "unresolved", "na")}
        judged = counts["pass"] + counts["fail"] + counts["unresolved"]
        failures = [f"{u.unit}: {u.reason}" if u.reason else u.unit for u in units if u.status in ("fail", "unresolved")]
        outcomes.append(RubricOutcome(id=rubric.id, counts=counts, pass_rate=counts["pass"] / judged if judged else None,
                                      failures=failures[:20]))
    return outcomes


def render_markdown(outcomes: list[RubricOutcome], rubrics: RubricSet) -> str:
    def cell(text: str) -> str:
        return text.replace("|", r"\|").replace("\n", " ")

    lines = ["# Rubric 通过率", "", "通过率 = pass / (pass + fail + unresolved)；na 不计入。只报告，不做门禁。", "",
             "| rubric | 问题 | 通过率 | pass | fail | unresolved |", "|---|---|---|---|---|---|"]
    for o in outcomes:
        rate = "-" if o.pass_rate is None else f"{o.pass_rate:.3f}"
        lines.append(f"| {o.id} | {cell(rubrics.by_id[o.id].question)} | {rate} | {o.counts['pass']} | "
                     f"{o.counts['fail']} | {o.counts['unresolved']} |")
    failing = [o for o in outcomes if o.failures]
    if failing:
        lines += ["", "## 失败与未决单元（每条 rubric 最多 20 个）", ""]
        for o in failing:
            lines += [f"### {o.id}", ""] + [f"- {cell(f)}" for f in o.failures] + [""]
    return "\n".join(lines) + "\n"


# --- Single source for schema-field wording ----------------------------------------------
# schema_fields.yaml is the only place that words each schema field. The LLM prompt, Jev
# instructions and the human review question all render from it.

FIELD_PLACEHOLDER = "{{FIELD_DEFINITIONS}}"


class SchemaField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    auxiliary: bool = False

    @field_validator("definition")
    @classmethod
    def normalized(cls, value: str) -> str:
        return " ".join(value.split())


class SchemaFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_question: str
    fields: list[SchemaField] = Field(min_length=1)

    @model_validator(mode="after")
    def check(self) -> SchemaFields:
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("duplicate field definitions")
        if "{field}" not in self.review_question:
            raise ValueError("review_question must contain {field}")
        return self


def load_schema_fields(path: str | Path | None = None) -> SchemaFields:
    return SchemaFields.model_validate(yaml.safe_load(Path(path or SCHEMA_FIELDS_PATH).read_text(encoding="utf-8")))


def schema_fields_hash(path: str | Path | None = None) -> dict[str, str]:
    return {"evaluation/schema_fields.yaml": hashlib.sha256(Path(path or SCHEMA_FIELDS_PATH).read_bytes()).hexdigest()}


def field_definitions(path: str | Path | None = None) -> dict[str, str]:
    return {f.name: f.definition for f in load_schema_fields(path).fields}


def render_field_definitions(path: str | Path | None = None) -> str:
    return "\n".join(f"- {f.name}: {f.definition}" for f in load_schema_fields(path).fields)


def render_schema_prompt(template: str, path: str | Path | None = None) -> str:
    if template.count(FIELD_PLACEHOLDER) != 1:
        raise ValueError(f"schema prompt must contain {FIELD_PLACEHOLDER} exactly once")
    return template.replace(FIELD_PLACEHOLDER, render_field_definitions(path))


def jev_instruction(field: str, rules: str, definitions: dict[str, str]) -> str:
    """Judge-specific framing around the shared definition."""
    return f"{rules}\nJudge ONLY `{field}`. {definitions[field]}"


def review_questions(path: str | Path | None = None) -> dict[str, str]:
    """Human annotation question per schema field: the shared template plus the shared definition."""
    spec = load_schema_fields(path)
    return {f.name: f"{spec.review_question.replace('{field}', f.name)} Criterion: {f.definition}" for f in spec.fields}
