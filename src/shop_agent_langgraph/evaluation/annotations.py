from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

from ..agents.eval.schemas import EvalReport, MatchingReport
from ..agents.eval.tools import validate_eval_report
from ..agents.eval.io import ITEM_FIELDS
from ..domain.criteria import CriteriaAttributeSet
from .cases import digest, seal_case, validate_case, write_json
from .langfuse_api import api, all_items

QUEUE_NAME = "market-dimension-review"
SCORE_NAME = "human_correct"
REVIEW_VERSION = "v1"


def review_tasks(case, report: EvalReport | None = None):
    gold = validate_case(case)
    if report is not None:
        report_gold = CriteriaAttributeSet.model_validate({"source_item_ids": report.source_item_ids,
            **{kind: [entry["item"] for ref, entry in report.items.items() if ref.startswith("gold:") and entry["kind"] == label]
               for kind, label in (("criteria", "criterion"), ("attributes", "attribute"))}})
        if report_gold != gold or report.gold_metadata.get("input_sha256", case["metadata"]["input_sha256"]) != case["metadata"]["input_sha256"]:
            raise ValueError("report does not belong to this frozen case/gold version")
    tasks = []

    def add(kind, ref, field, inp, output):
        identity = dict(input_sha256=case["metadata"]["input_sha256"],
                        gold_sha256=case["metadata"]["expected_output_sha256"],
                        task_kind=kind, ref=ref, field=field, input=inp, output=output,
                        review_version=REVIEW_VERSION)
        tasks.append(dict(id=digest(identity), task_kind=kind, ref=ref, field=field,
                          input=inp, output=output, metadata={k: v for k, v in identity.items() if k not in {"input", "output"}}))

    for kind in ("criteria", "attributes"):
        for item in case["gold_annotation"][kind]:
            add("gold_item", item["id"], None,
                {"question": "Is this complete gold definition/schema and its product evidence mapping correct? Correct the entire item if needed.",
                 "kind": kind, "source_item_ids": case["input"]["source_item_ids"], "evidence": item.get("evidence", [])}, item)
    add("gold_collection", "collection", None,
        {"question": "Is this collection complete and free of duplicate or improperly merged dimensions? Correct the full collection if needed.",
         "source_item_ids": case["input"]["source_item_ids"],
         "product_contexts": case["input"]["product_contexts"]}, case["expected_output"])
    if report is not None:
        matching = {key: getattr(report, key) for key in ("match", "missing", "extra")}
        matching = MatchingReport(**matching).model_dump(mode="json")
        add("matching", "collection", None,
            {"question": "Are all match/missing/extra assignments correct? Corrections must cover every reference exactly once.",
             "items": {r: {k: e["item"][k] for k in ("name", "description")} for r, e in report.items.items()}}, matching)
        groups = {ref: group for group in report.match for ref in group.gold_ids}
        for review in report.reviews:
            for score in review.fields:
                if score.field == "granularity":
                    continue  # deterministic; correcting matching recomputes this
                group = groups[review.gold_ref]
                add("schema_field", review.gold_ref, score.field,
                    {"question": f"Does the actual representation preserve the gold dimension's `{score.field}` semantics? Score the field itself, not agreement with an automatic judge.",
                     "gold": report.items[review.gold_ref], "actual": {r: report.items[r] for r in group.actual_ids}},
                    {"field": score.field, "actual": {r: report.items[r]["item"].get(score.field) if score.field != "kind" else report.items[r]["kind"] for r in group.actual_ids}})
                decision = next((d for d in report.judge_details.get("decisions", [])
                                 if d["gold_ref"] == review.gold_ref and d["field"] == score.field), None)
                tasks[-1]["automatic"] = {"llm": score.score, "jev": decision}
    return tasks


def ensure_queue():
    configs = [c for c in all_items("score-configs", "list") if c["name"] == SCORE_NAME and not c.get("isArchived")]
    if len(configs) > 1 or (configs and configs[0]["dataType"] != "BOOLEAN"):
        raise ValueError("human_correct has ambiguous or incompatible score configs")
    config = configs[0] if configs else api("score-configs", "create", body={
        "name": SCORE_NAME, "dataType": "BOOLEAN", "description": "1 means the specific review question is correct; 0 means incorrect. Leave unresolved tasks pending."})
    queues = [q for q in all_items("annotation-queues", "list") if q["name"] == QUEUE_NAME]
    if len(queues) > 1:
        raise ValueError("review queue name is ambiguous")
    queue = queues[0] if queues else api("annotation-queues", "create", body={
        "name": QUEUE_NAME, "scoreConfigIds": [config["id"]], "description": "Fixed-product gold, matching and schema field review. Filter by task_kind."})
    if config["id"] not in queue["scoreConfigIds"]:
        raise ValueError("existing review queue does not use the expected score config")
    return queue, config


def publish_tasks(tasks, *, client, ledger_path):
    queue, config = ensure_queue()
    ledger_path = Path(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    existing = {item["objectId"] for item in all_items("annotation-queues", "list-items", queue["id"])}
    for task in tasks:
        entry = ledger.get(task["id"])
        if entry is None:
            with client.start_as_current_observation(name=f"review-{task['task_kind']}", input=task["input"],
                    output=task["output"], metadata={**task["metadata"], "review_task_id": task["id"]}) as span:
                entry = {"task": task, "trace_id": span.trace_id, "observation_id": span.id,
                         "queue_id": queue["id"], "config_id": config["id"]}
            ledger[task["id"]] = entry
            write_json(ledger_path, ledger)  # persist before remote enqueue, allowing retry
            client.flush()
        if entry["queue_id"] != queue["id"]:
            raise ValueError("review ledger belongs to a different queue")
        if entry["observation_id"] not in existing:
            api("annotation-queues", "create-item", queue["id"], body={
                "objectId": entry["observation_id"], "objectType": "OBSERVATION"})
            existing.add(entry["observation_id"])
    return ledger


def collect_reviews(ledger):
    queue_ids = {entry["queue_id"] for entry in ledger.values()}
    if len(queue_ids) != 1:
        raise ValueError("review ledger must belong to one queue")
    queue_id = next(iter(queue_ids))
    completed = {item["objectId"] for item in all_items("annotation-queues", "list-items", queue_id, "--status", "COMPLETED")}
    labels = all_items("scores", "list", "--queue-id", queue_id, "--name", SCORE_NAME,
                       "--source", "ANNOTATION", "--fields", "subject,details,annotation")
    by_subject = {}
    for score in labels:
        subject = score.get("subject", {})
        if subject.get("kind") == "observation":
            by_subject.setdefault(subject["id"], []).append(score)
    reviews = []
    for task_id, entry in ledger.items():
        if entry["observation_id"] not in completed:
            continue
        scores = by_subject.get(entry["observation_id"], [])
        if len(scores) != 1 or scores[0].get("configId") != entry["config_id"]:
            raise ValueError(f"task {task_id} needs exactly one human label under the expected config")
        score = scores[0]
        if score["dataType"] != "BOOLEAN" or type(score["value"]) is not bool:
            raise ValueError("human label must be BOOLEAN")
        corrections = all_items("scores", "list", "--trace-id", entry["trace_id"],
            "--observation-id", entry["observation_id"], "--data-type", "CORRECTION", "--fields", "subject,details,annotation")
        if len(corrections) > 1:
            raise ValueError("task has ambiguous corrections")
        correction = json.loads(corrections[0]["value"]) if corrections else None
        reviews.append({**entry, "task_id": task_id, "human_score": int(score["value"]),
                        "comment": score.get("comment", ""), "correction": correction,
                        "score_id": score["id"], "reviewer": score.get("authorUserId")})
    return reviews


def attach_judge_scores(reviews, client):
    """Publish automated comparators only after the independent human label exists."""
    for review in reviews:
        automatic = review["task"].get("automatic", {})
        scores = []
        if "llm" in automatic:
            scores.append(("llm_correct", automatic["llm"], "BOOLEAN"))
        jev = automatic.get("jev")
        if jev is not None and jev.get("source") == "jev":
            scores.append(("jev_choice", jev["choice"], "CATEGORICAL"))
            scores.append(("jev_confidence", jev["confidence"], "NUMERIC"))
            if jev["candidate_score"] is not None:
                scores.append(("jev_correct", jev["candidate_score"], "BOOLEAN"))
        for name, value, data_type in scores:
            client.create_score(name=name, value=value, data_type=data_type,
                trace_id=review["trace_id"], observation_id=review["observation_id"],
                score_id=digest({"task": review["task_id"], "name": name}),
                metadata={"review_task_id": review["task_id"], "field": review["task"]["field"],
                          "model": jev.get("model") if jev else None})
    client.flush()


def export_candidates(case, reviews, directory, report=None):
    validate_reviews(case, reviews, report)
    directory = Path(directory)
    annotation = deepcopy(case["gold_annotation"])
    by_ref = {r["task"]["ref"]: r for r in reviews if r["task"]["task_kind"] == "gold_item"}
    for kind in ("criteria", "attributes"):
        annotation[kind] = [by_ref[item["id"]]["correction"]
                           if item["id"] in by_ref and by_ref[item["id"]].get("correction") is not None else item
                           for item in annotation[kind]]
    collection_reviews = [r for r in reviews if r["task"]["task_kind"] == "gold_collection"]
    if len(collection_reviews) > 1:
        raise ValueError("ambiguous collection reviews")
    if collection_reviews and collection_reviews[0].get("correction") is not None:
        corrected = collection_reviews[0]["correction"]
        for kind in ("criteria", "attributes"):
            old = {item["id"]: item for item in annotation[kind]}
            annotation[kind] = [{**old.get(item["id"], {"evidence_item_ids": [], "evidence": []}), **item}
                                for item in corrected[kind]]
    annotation["annotation_metadata"].update(status="draft_pending_human_review",
        parent_gold_sha256=case["metadata"]["expected_output_sha256"],
        human_review_score_ids=[r["score_id"] for r in reviews],
        authoring="Original draft plus exported human corrections; final approval is still pending")
    canonical = {"source_item_ids": annotation["source_item_ids"],
                 **{kind: [{k: v for k, v in item.items() if k in ITEM_FIELDS} for item in annotation[kind]]
                    for kind in ("criteria", "attributes")}}
    CriteriaAttributeSet.model_validate(canonical)
    gold_path = write_json(directory / "gold-candidate.json", annotation)
    candidate = deepcopy(case)
    candidate.update(expected_output=canonical, gold_annotation=annotation)
    candidate["metadata"].update(status="draft_pending_human_review", gold_path=str(gold_path.resolve()),
        gold_sha256=hashlib.sha256(gold_path.read_bytes()).hexdigest())
    write_json(directory / "case-candidate.json", seal_case(candidate))
    calibration = []
    for review in reviews:
        task = review["task"]
        if task["task_kind"] == "schema_field":
            calibration.append({"input": task["input"], "field": task["field"],
                "expected_output": review["human_score"], "comment": review.get("comment", ""),
                "task_id": task["id"], "metadata": task["metadata"]})
    write_json(directory / "jev-calibration.json", calibration)
    write_json(directory / "human-reviews.json", reviews)
    return gold_path


def validate_reviews(case, reviews, report=None):
    """Validate exports without overwriting gold or treating completed tasks as approval."""
    gold = validate_case(case)
    seen = set()
    for review in reviews:
        task = review["task"]
        if digest({**task["metadata"], "input": task["input"], "output": task["output"]}) != task["id"]:
            raise ValueError("review task content/hash changed")
        if task["id"] in seen or type(review["human_score"]) is not int or review["human_score"] not in (0, 1):
            raise ValueError("duplicate review or non-binary human score")
        seen.add(task["id"])
        if task["metadata"]["input_sha256"] != case["metadata"]["input_sha256"] or task["metadata"]["gold_sha256"] != case["metadata"]["expected_output_sha256"]:
            raise ValueError("review belongs to another case version")
        correction = review.get("correction")
        if correction is None:
            continue
        kind = task["task_kind"]
        if kind == "gold_item":
            item_kind = task["input"]["kind"]
            CriteriaAttributeSet.model_validate({"source_item_ids": gold.source_item_ids,
                "criteria": [{k: v for k, v in correction.items() if k in ITEM_FIELDS}] if item_kind == "criteria" else [],
                "attributes": [{k: v for k, v in correction.items() if k in ITEM_FIELDS}] if item_kind == "attributes" else []})
            if correction["id"] != task["ref"] or not set(correction.get("evidence_item_ids", [])) <= set(gold.source_item_ids):
                raise ValueError("corrected gold item must preserve ID and use fixed product evidence")
            for evidence in correction.get("evidence", []):
                text = case["input"]["product_contexts"].get(evidence["item_id"], "")
                if not evidence.get("quote", "").strip() or evidence["quote"] not in text:
                    raise ValueError("corrected evidence must quote the frozen product context")
        elif kind == "gold_collection":
            corrected = CriteriaAttributeSet.model_validate(correction)
            if corrected.source_item_ids != gold.source_item_ids:
                raise ValueError("collection correction changed product scope")
        elif kind == "matching":
            if report is None:
                raise ValueError("matching corrections require the original report")
            actual = CriteriaAttributeSet.model_validate({"source_item_ids": gold.source_item_ids,
                **{kind: [e["item"] for r, e in report.items.items() if r.startswith("actual:") and e["kind"] == label]
                   for kind, label in (("criteria", "criterion"), ("attributes", "attribute"))}})
            validate_eval_report(MatchingReport.model_validate(correction), gold, actual)
        elif kind == "schema_field":
            raise ValueError("schema field tasks use the human 0/1 label and comment, not corrected output")
    return reviews
