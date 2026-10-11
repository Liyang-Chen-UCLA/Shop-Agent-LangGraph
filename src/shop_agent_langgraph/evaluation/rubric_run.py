"""Compute and persist rubric pass rates for saved fixed-product experiment runs."""
from __future__ import annotations

import json
from pathlib import Path

from langfuse import Evaluation

from ..agents.eval.schemas import EvalReport
from .cases import write_json
from .rubric import (GoldOverlay, RubricOutcome, RubricSet, UnitResult, load_overlay, load_rubrics, render_markdown,
                     rubric_file_hashes, summarize)
from .rubric_checks import CheckContext, run_module

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OVERLAY = ROOT / "eval/cases/gamepad/rubric.yaml"


def compute_trial(case: dict, output: dict, report: EvalReport | dict | None, overlay: GoldOverlay,
                  rubrics: RubricSet | None = None) -> dict[str, list[UnitResult]]:
    """Research + aggregation rubrics for one delivered run.

    Research rubrics need the per-product research audit (absent when an existing
    delivery is replayed); aggregation rubrics need the EvalReport.
    """
    rubrics = rubrics or load_rubrics()
    context = CheckContext(overlay=overlay, rubrics=rubrics)
    results: dict[str, list[UnitResult]] = {}
    contexts = case["input"]["product_contexts"]
    research = {e["item_id"]: {"item_id": e["item_id"], "context": contexts[e["item_id"]], "result": e,
                               "reference": {"relevance": "relevant"}}
                for e in output.get("audit", []) if e.get("stage") == "fixed_research" and e["item_id"] in contexts}
    if research:
        results.update(run_module("research", research, context=context))
    if report is not None:
        sample = {"report": report, "input_item_ids": case["input"]["source_item_ids"],
                  "actual": {"source_item_ids": output["item_ids"], "criteria": output["criteria"],
                             "attributes": output["attributes"]}}
        results.update(run_module("aggregation", {"delivery": sample}, context=context))
    return results


def write_rubric_artifacts(directory: str | Path, trial: dict[str, list[UnitResult]], case: dict,
                           overlay: GoldOverlay, rubrics: RubricSet | None = None) -> list[RubricOutcome]:
    rubrics = rubrics or load_rubrics()
    outcomes = summarize(rubrics, trial)
    directory = Path(directory)
    write_json(directory / "rubric.json", {
        "rubric_hashes": rubric_file_hashes(), "input_sha256": case["metadata"]["input_sha256"],
        "overlay_status": overlay.status,
        "trial": {rid: [u.model_dump(mode="json") for u in units] for rid, units in trial.items()},
        "summary": [o.model_dump(mode="json") for o in outcomes]})
    (directory / "rubric.md").write_text(render_markdown(outcomes, rubrics), encoding="utf-8")
    return outcomes


def rubric_scores(outcomes: list[RubricOutcome]) -> list[Evaluation]:
    return [Evaluation(name=f"rubric.{o.id}", value=float(o.pass_rate), data_type="NUMERIC")
            for o in outcomes if o.pass_rate is not None]


def load_run_trial(run_dir: str | Path, overlay: GoldOverlay, rubrics: RubricSet) -> dict:
    """Read a run's saved rubric trial, or recompute it when missing or scored with other rubric files."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) if (run_dir / "manifest.json").exists() else {}
    if manifest and manifest.get("status") != "completed":
        raise ValueError(f"{run_dir.name}: run did not complete")
    case = json.loads((run_dir / "case.json").read_text(encoding="utf-8"))
    saved = run_dir / "rubric.json"
    if saved.exists():
        data = json.loads(saved.read_text(encoding="utf-8"))
        if data["rubric_hashes"] == rubric_file_hashes():
            trial = {rid: [UnitResult.model_validate(u) for u in units] for rid, units in data["trial"].items()}
            return dict(trial=trial, case=case, source="rubric.json")
    output = json.loads((run_dir / "market.json").read_text(encoding="utf-8"))
    report_path = run_dir / "report.json"
    report = EvalReport.model_validate_json(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    return dict(trial=compute_trial(case, output, report, overlay, rubrics), case=case, source="recomputed")


def load_runs(run_dirs: list[str | Path], overlay_path: str | Path | None = None) -> list[dict]:
    """Load several runs of the same case and gold version with their rubric trials."""
    if not run_dirs:
        raise ValueError("need at least one run directory")
    overlay = load_overlay(overlay_path or DEFAULT_OVERLAY)
    rubrics = load_rubrics()
    runs = [load_run_trial(d, overlay, rubrics) for d in run_dirs]
    if len({r["case"]["metadata"]["input_sha256"] for r in runs}) != 1 or             len({r["case"]["metadata"]["expected_output_sha256"] for r in runs}) != 1:
        raise ValueError("runs belong to different cases or gold versions")
    return runs
