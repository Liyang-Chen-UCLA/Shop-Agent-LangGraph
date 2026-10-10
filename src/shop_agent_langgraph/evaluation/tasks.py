"""Hand-written rubric tasks (eval/tasks/<module>/*.json) and their sample adapters."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .rubric import MODULE_PREFIXES, RubricSet, load_rubrics

ROOT = Path(__file__).resolve().parents[3]
TASKS_DIR = ROOT / "eval/tasks"


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    module: str
    input: dict
    reference: dict
    rubric_ids: list[str] = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


def load_tasks(module: str, *, directory: str | Path | None = None, rubrics: RubricSet | None = None) -> list[Task]:
    """Load one module's tasks; every referenced rubric must exist in the same module."""
    if module not in MODULE_PREFIXES:
        raise ValueError(f"unknown module {module!r}")
    rubrics = rubrics or load_rubrics()
    known = {r.id for r in rubrics.module(module)}
    tasks = []
    for path in sorted((Path(directory) if directory else TASKS_DIR / module).glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) != {"input", "reference", "rubric_ids", "metadata"}:
            raise ValueError(f"{path.name}: tasks need exactly input, reference, rubric_ids, metadata")
        task = Task(id=path.stem, module=module, **raw)
        if len(task.rubric_ids) != len(set(task.rubric_ids)) or not set(task.rubric_ids) <= known:
            raise ValueError(f"{path.name}: rubric_ids must be distinct {module} rubrics")
        tasks.append(task)
    return tasks


def gold_approved(task: Task) -> bool:
    return task.metadata.get("reference_status") == "approved"


def intent_sample(task: Task, output: dict) -> dict:
    return {"input": task.input["message"], "output": output, "reference": task.reference}


def route_sample(task: Task, output: dict) -> dict:
    return {"input": task.input["product"], "output": output, "reference": task.reference}


def e2e_sample(task: Task, trace: dict) -> dict:
    return {"input": task.input["message"], "reference": task.reference, "trace": trace}


def research_sample(task: Task, result: dict, *, context: str | None = None, root: Path = ROOT) -> dict:
    """Reference the gold by path: product text comes from the frozen store, evidence from gold."""
    item_id = task.reference["item_id"]
    if context is None:
        from ..domain.market_env import get_product_raw_text
        context = get_product_raw_text(item_id)
    gold = json.loads((root / task.reference["gold"]).read_text(encoding="utf-8"))
    evidence = [{"dimension_id": dim["id"], "name": dim["name"], "quote": ev["quote"]}
                for kind in ("criteria", "attributes") for dim in gold[kind]
                for ev in dim.get("evidence", []) if ev["item_id"] == item_id]
    return {"item_id": item_id, "context": context, "result": result,
            "reference": {"relevance": task.reference.get("relevance", "relevant"), "gold_evidence": evidence}}
