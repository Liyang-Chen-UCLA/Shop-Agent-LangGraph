"""Compute, persist and gate rubric results for saved fixed-product experiment runs."""
from __future__ import annotations

import json
from pathlib import Path

from langfuse import Evaluation

from ..agents.eval.schemas import EvalReport
from .cases import write_json
from .rubric import (GateReport, GoldOverlay, RubricSet, UnitResult, evaluate_gate, load_overlay, load_rubrics,
                     render_gate_markdown, rubric_file_hashes)
from .rubric_checks import CheckContext, run_module

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OVERLAY = ROOT / "eval/cases/gamepad/rubric.yaml"
RUN_MODULES = ("research", "aggregation")


def gold_approved(case: dict, overlay: GoldOverlay) -> bool:
    return case["metadata"].get("status") == "approved" and overlay.approved


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
                           overlay: GoldOverlay, rubrics: RubricSet | None = None) -> GateReport:
    rubrics = rubrics or load_rubrics()
    approved = gold_approved(case, overlay)
    report = evaluate_gate(rubrics, [trial], gold_approved=approved, modules=list(RUN_MODULES))
    directory = Path(directory)
    write_json(directory / "rubric.json", {
        "rubric_hashes": rubric_file_hashes(), "input_sha256": case["metadata"]["input_sha256"],
        "overlay_status": overlay.status, "gold_approved": approved,
        "trial": {rid: [u.model_dump(mode="json") for u in units] for rid, units in trial.items()},
        "gate": report.model_dump(mode="json")})
    (directory / "rubric.md").write_text(render_gate_markdown(report), encoding="utf-8")
    return report


def rubric_scores(report: GateReport) -> list[Evaluation]:
    scores = [Evaluation(name=f"rubric.{o.id}", value=float(o.pass_rate), data_type="NUMERIC", comment=o.verdict)
              for o in report.outcomes if o.pass_rate is not None]
    scores.append(Evaluation(name="rubric.gate", value=report.conclusion, data_type="CATEGORICAL",
                             comment=f"blockers={report.blockers}; needs_review={report.reviews}"))
    return scores


def load_run_trial(run_dir: str | Path, overlay: GoldOverlay, rubrics: RubricSet) -> dict:
    """Read a run's saved rubric trial, or recompute it from market.json/report.json/case.json."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) if (run_dir / "manifest.json").exists() else {}
    if manifest and manifest.get("status") != "completed":
        raise ValueError(f"{run_dir.name}: run did not complete")
    case = json.loads((run_dir / "case.json").read_text(encoding="utf-8"))
    saved = run_dir / "rubric.json"
    if saved.exists():
        data = json.loads(saved.read_text(encoding="utf-8"))
        trial = {rid: [UnitResult.model_validate(u) for u in units] for rid, units in data["trial"].items()}
        return dict(trial=trial, case=case, hashes=data["rubric_hashes"], source="rubric.json")
    output = json.loads((run_dir / "market.json").read_text(encoding="utf-8"))
    report_path = run_dir / "report.json"
    report = EvalReport.model_validate_json(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    return dict(trial=compute_trial(case, output, report, overlay, rubrics), case=case,
                hashes=rubric_file_hashes(), source="recomputed")


def gate_runs(run_dirs: list[str | Path], overlay_path: str | Path | None = None) -> tuple[GateReport, list[dict]]:
    """Treat several run directories as trials of the same case and gate them together."""
    if not run_dirs:
        raise ValueError("gate needs at least one run directory")
    overlay = load_overlay(overlay_path or DEFAULT_OVERLAY)
    rubrics = load_rubrics()
    runs = [load_run_trial(d, overlay, rubrics) for d in run_dirs]
    if len({json.dumps(r["hashes"], sort_keys=True) for r in runs}) != 1:
        raise ValueError("cannot gate runs scored with different rubric files")
    if len({r["case"]["metadata"]["input_sha256"] for r in runs}) != 1 or \
            len({r["case"]["metadata"]["expected_output_sha256"] for r in runs}) != 1:
        raise ValueError("cannot gate runs of different cases or gold versions")
    approved = gold_approved(runs[0]["case"], overlay)
    modules = [m for m in RUN_MODULES if any(x.id in r["trial"] for r in runs for x in rubrics.module(m))]
    return evaluate_gate(rubrics, [r["trial"] for r in runs], gold_approved=approved, modules=modules or list(RUN_MODULES)), runs
