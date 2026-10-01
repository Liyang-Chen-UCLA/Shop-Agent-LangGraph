"""Prepare, run, review and export the fixed-product evaluation lifecycle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from shop_agent_langgraph.agents.eval.jev import JevJudge
from shop_agent_langgraph.agents.eval.schemas import EvalReport
from shop_agent_langgraph.core.tracing import get_tracing_client, load_environment
from shop_agent_langgraph.evaluation.annotations import attach_judge_scores, collect_reviews, export_candidates, publish_tasks, review_tasks
from shop_agent_langgraph.evaluation.cases import load_case, prepare_case, write_json
from shop_agent_langgraph.evaluation.experiment import run_experiment
from shop_agent_langgraph.evaluation.calibration import calibrate, compare_runs
from shop_agent_langgraph.evaluation.datasets import DATASET_NAME, case_from_dataset, dataset_draft, publish_dataset

ROOT = Path(__file__).resolve().parents[1]


def tracing_client(required):
    client = get_tracing_client() if required else None
    if required and (client is None or not client.auth_check()):
        raise RuntimeError("Langfuse is disabled, unconfigured or failed authentication")
    return client


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--gold", type=Path, default=ROOT / "eval/cases/gamepad/gold.json")
    prepare.add_argument("--output", type=Path, default=ROOT / ".cache/eval/gamepad/case.json")
    run = sub.add_parser("run")
    run.add_argument("--case", type=Path, default=ROOT / ".cache/eval/gamepad/case.json")
    run.add_argument("--output", type=Path)
    run.add_argument("--name", default="baseline")
    run.add_argument("--judge", choices=["llm", "jev-shadow"], default="jev-shadow")
    run.add_argument("--actual", type=Path, help="Explicitly replay an existing delivery; does not rerun Market")
    run.add_argument("--langfuse", action="store_true", help="Publish the experiment and scores")
    run.add_argument("--dataset")
    run.add_argument("--dataset-version", help="Exact Langfuse dataset version timestamp")
    dataset = sub.add_parser("publish-case")
    dataset.add_argument("--case", type=Path, default=ROOT / ".cache/eval/gamepad/case.json")
    dataset.add_argument("--name", default=DATASET_NAME)
    dataset.add_argument("--output", type=Path, default=ROOT / ".cache/eval/gamepad/dataset-draft.json")
    dataset.add_argument("--apply", action="store_true", help="Publish the reviewed draft to Langfuse")
    review = sub.add_parser("review")
    review.add_argument("--case", type=Path, default=ROOT / ".cache/eval/gamepad/case.json")
    review.add_argument("--report", type=Path)
    review.add_argument("--output", type=Path, default=ROOT / ".cache/eval/gamepad/review-tasks.json")
    review.add_argument("--kinds", nargs="+", choices=["gold_item", "gold_collection", "matching", "schema_field"])
    review.add_argument("--limit", type=int)
    publish = sub.add_parser("publish-review")
    publish.add_argument("--tasks", type=Path, required=True)
    publish.add_argument("--ledger", type=Path, default=ROOT / ".cache/eval/gamepad/review-ledger.json")
    export = sub.add_parser("export-review")
    export.add_argument("--case", type=Path, default=ROOT / ".cache/eval/gamepad/case.json")
    export.add_argument("--ledger", type=Path, default=ROOT / ".cache/eval/gamepad/review-ledger.json")
    export.add_argument("--report", type=Path)
    export.add_argument("--output", type=Path, default=ROOT / ".cache/eval/gamepad/review-export")
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--labels", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--langfuse", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        case = prepare_case(args.gold)
        write_json(args.output, case)
        print(f"Prepared {len(case['input']['source_item_ids'])} products; gold status: {case['metadata'].get('status')}")
        print(args.output)
    elif args.command == "run":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or ROOT / ".cache/eval/gamepad/runs" / stamp
        jev = None
        if args.judge == "jev-shadow":
            threshold = os.getenv("JEV_MIN_CONFIDENCE", "").strip()
            jev = JevJudge(model=os.getenv("JEV_MODEL", "jev-1.13.0"),
                           min_confidence=float(threshold) if threshold else None)
        client = tracing_client(args.langfuse)
        dataset = None
        if args.dataset:
            if client is None or not args.dataset_version:
                parser.error("--dataset requires --langfuse and an explicit --dataset-version")
            dataset = client.get_dataset(args.dataset,
                version=datetime.fromisoformat(args.dataset_version.replace("Z", "+00:00")))
            case = case_from_dataset(dataset)
            case["metadata"].update(dataset_name=args.dataset, dataset_version=args.dataset_version)
        else:
            case = load_case(args.case)
        manifest = run_experiment(case, output, client=client, dataset=dataset,
                                  jev=jev, actual_path=args.actual, run_name=f"{args.name}-{stamp}")
        print(f"Completed: {output}")
        print(f"Experiment: {manifest.get('experiment_url') or manifest.get('experiment_id', 'local')}")
    elif args.command == "publish-case":
        case = load_case(args.case)
        write_json(args.output, publish_dataset(case, args.name) if args.apply else dataset_draft(case, args.name))
        print(f"{'Published dataset' if args.apply else 'Prepared dataset draft'}: {args.output}")
    elif args.command == "review":
        report = EvalReport.model_validate_json(args.report.read_text(encoding="utf-8")) if args.report else None
        tasks = review_tasks(load_case(args.case), report)
        if args.kinds:
            tasks = [t for t in tasks if t["task_kind"] in args.kinds]
        if args.limit is not None:
            if args.limit < 1:
                parser.error("--limit must be positive")
            tasks = tasks[:args.limit]
        write_json(args.output, tasks)
        print(f"Prepared {len(tasks)} review tasks: {args.output}")
    elif args.command == "publish-review":
        tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
        ledger = publish_tasks(tasks, client=tracing_client(True), ledger_path=args.ledger)
        print(f"Review ledger: {args.ledger}; tracked tasks: {len(ledger)}")
    elif args.command == "export-review":
        report = EvalReport.model_validate_json(args.report.read_text(encoding="utf-8")) if args.report else None
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
        reviews = collect_reviews(ledger)
        export_candidates(load_case(args.case), reviews, args.output, report)
        attach_judge_scores(reviews, tracing_client(True))
        print(f"Exported {len(reviews)} completed reviews: {args.output}")
    elif args.command == "calibrate":
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        result = calibrate(labels, JevJudge(model=os.getenv("JEV_MODEL", "jev-1.13.0")),
                           args.output, client=tracing_client(args.langfuse))
        print(json.dumps(result, ensure_ascii=False))
    else:
        write_json(args.output, compare_runs(args.baseline, args.candidate))
        print(args.output)


if __name__ == "__main__":
    main()
