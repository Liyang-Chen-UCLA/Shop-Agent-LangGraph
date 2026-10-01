"""Prepare an exact Dataset draft and publish it only via an explicit command."""
from __future__ import annotations

from .cases import digest, validate_case
from .langfuse_api import api, all_items

DATASET_NAME = "market/fixed-products/gamepad"


def dataset_draft(case, name=DATASET_NAME):
    validate_case(case)
    return {"dataset": {"name": name, "description": "Fixed independent-controller snapshot; evaluate Research and Market delivery, not search or screening.",
                        "metadata": {"test_level": "market-fixed-products", "case_family": "independent-gamepad"}},
            "item": {"datasetName": name, "id": digest({"dataset": name, "case_family": "independent-gamepad"}),
                     "input": case["input"], "expectedOutput": case["expected_output"],
                     "metadata": {**case["metadata"], "gold_annotation": case["gold_annotation"]}}}


def publish_dataset(case, name=DATASET_NAME):
    draft = dataset_draft(case, name)
    found = [d for d in all_items("datasets", "list") if d["name"] == name]
    if found:
        if len(found) != 1 or found[0].get("metadata", {}).get("case_family") != "independent-gamepad":
            raise ValueError("existing dataset name belongs to a different case family")
        dataset = found[0]
    else:
        dataset = api("datasets", "create", body=draft["dataset"])
    item = api("dataset-items", "create", body=draft["item"])
    return {"dataset": dataset, "item": item,
            "version": item.get("updatedAt") or item.get("createdAt")}


def case_from_dataset(dataset):
    if len(dataset.items) != 1:
        raise ValueError("this fixed-scope MVP expects exactly one dataset item")
    item = dataset.items[0]
    metadata = dict(item.metadata)
    annotation = metadata.pop("gold_annotation")
    case = {"input": item.input, "expected_output": item.expected_output,
            "metadata": metadata, "gold_annotation": annotation}
    validate_case(case)
    return case
