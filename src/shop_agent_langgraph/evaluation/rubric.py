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
