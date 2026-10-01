from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langfuse import Evaluation

from ..agents.eval.schemas import FieldScore, ItemReview, SchemaReview
from .cases import digest, write_json


def calibrate(samples, jev, directory, *, client=None):
    if not samples or len({s["task_id"] for s in samples}) != len(samples):
        raise ValueError("calibration needs nonempty, distinct reviewed samples")
    if any(type(s["expected_output"]) is not int or s["expected_output"] not in (0, 1) for s in samples):
        raise ValueError("calibration labels must be binary integers")
    jev.preflight()
    jev.trace_client = client
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    decisions = []

    async def task(*, item, **kwargs):
        field = item["metadata"]["field"]
        state = item["input"]
        # Never pass the human label to the model or use it as baseline scoring.
        baseline = SchemaReview(items=[ItemReview(gold_ref="gold:0", fields=[
            FieldScore(field=field, score=0, reason="Calibration placeholder, not provided to Jev")])])
        result = await jev.review([dict(gold={"gold:0": state["gold"]}, actual=state["actual"],
                                      required_fields={"gold:0": [field]})], baseline)
        decision = {**result["decisions"][0], "task_id": item["metadata"]["task_id"]}
        decisions.append(decision)
        return decision

    def evaluator(*, output, expected_output, **kwargs):
        return Evaluation(name="judge_output_correct", value=int(output["candidate_score"] == expected_output),
                          data_type="BOOLEAN", comment="Unclear judgments count as unresolved, not correct")

    data = [dict(input={"gold": s["input"]["gold"], "actual": s["input"]["actual"]},
                 expected_output=s["expected_output"], metadata={"field": s["field"], "task_id": s["task_id"]}) for s in samples]
    manifest = {"status": "running", "model": jev.model, "samples_sha256": digest(data), "approved": False}
    try:
        if client:
            result = client.run_experiment(name="jev-schema-calibration", data=data, task=task,
                                            evaluators=[evaluator], max_concurrency=1,
                                            metadata={"model": jev.model, "samples_sha256": manifest["samples_sha256"]})
            manifest["experiment_id"] = result.experiment_id
            manifest["experiment_url"] = result.dataset_run_url
        else:
            async def local():
                for item in data:
                    await task(item=item)
            asyncio.run(local())
        if len(decisions) != len(samples):
            raise RuntimeError("calibration did not finish every sample")
        labels = {s["task_id"]: s["expected_output"] for s in samples}
        for decision in decisions:
            decision["human_score"] = labels[decision["task_id"]]
            decision["agrees"] = decision["candidate_score"] == decision["human_score"]
        fields = {d["field"] for d in decisions}
        manifest["by_field"] = {field: {
            "count": sum(d["field"] == field for d in decisions),
            "correct": sum(d["field"] == field and d["agrees"] for d in decisions),
            "unclear": sum(d["field"] == field and d["candidate_score"] is None for d in decisions),
            "positive_labels": sum(d["field"] == field and d["human_score"] == 1 for d in decisions),
            "negative_labels": sum(d["field"] == field and d["human_score"] == 0 for d in decisions),
        } for field in fields}
        manifest["status"] = "completed"
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "decisions.json", decisions)
        if client:
            client.flush()
    return manifest


def compare_runs(baseline_path, candidate_path):
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if baseline["status"] != "completed" or candidate["status"] != "completed":
        raise ValueError("comparison requires completed experiments")
    for key in ("input_sha256", "expected_output_sha256", "judge_mode", "judge_model", "jev_model"):
        if baseline["metadata"].get(key) != candidate["metadata"].get(key):
            raise ValueError(f"cannot compare runs with different {key}")
    for name, value in baseline["metadata"]["prompt_hashes"].items():
        if name.startswith("agents/eval/") and candidate["metadata"]["prompt_hashes"].get(name) != value:
            raise ValueError("cannot compare runs judged with different evaluation prompts")
    for name, value in baseline["metadata"].get("code_hashes", {}).items():
        if (name.startswith("agents/eval/") or name == "domain/criteria.py") and candidate["metadata"].get("code_hashes", {}).get(name) != value:
            raise ValueError("cannot compare runs judged with different evaluation code")
    b, c = baseline["metrics"], candidate["metrics"]
    return {"counts": {key: {"baseline": b[key], "candidate": c[key], "delta": c[key] - b[key]}
                       for key in ("missing_count", "extra_count", "classification_error_count", "granularity_error_count")},
            "fields": {key: {"baseline": value, "candidate": c["field_scores"][key],
                               "delta": None if value is None or c["field_scores"][key] is None else c["field_scores"][key] - value}
                       for key, value in b["field_scores"].items()}}
