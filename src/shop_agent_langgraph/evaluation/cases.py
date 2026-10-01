from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..agents.eval.io import ITEM_FIELDS, load_gold
from ..agents.eval.tools import source_references
from ..domain.criteria import CriteriaAttributeSet
from ..domain.market_env import get_product, products_path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_case(gold_path):
    gold, metadata = load_gold(gold_path)
    source_references(gold, gold)
    annotation = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    contexts = {item_id: get_product(item_id)["context_text"] for item_id in gold.source_item_ids}
    payload = dict(input={"target": "独立手柄", "source_item_ids": gold.source_item_ids,
                          "product_contexts": contexts}, expected_output=gold.model_dump(mode="json"),
                   metadata={**metadata, "test_level": "market-fixed-products",
                             "external_search": False,
                             "products_sha256": hashlib.sha256(products_path().read_bytes()).hexdigest()},
                   gold_annotation=annotation)
    return seal_case(payload)


def seal_case(case):
    case["metadata"]["input_sha256"] = digest(case["input"])
    case["metadata"]["expected_output_sha256"] = digest(case["expected_output"])
    case["metadata"]["annotation_sha256"] = digest(case["gold_annotation"])
    return case


def validate_case(case):
    gold = CriteriaAttributeSet.model_validate(case["expected_output"])
    source_references(gold, gold)
    annotation = case["gold_annotation"]
    annotated_gold = CriteriaAttributeSet.model_validate({"source_item_ids": annotation["source_item_ids"],
        **{kind: [{k: v for k, v in item.items() if k in ITEM_FIELDS} for item in annotation[kind]]
           for kind in ("criteria", "attributes")}})
    if annotated_gold != gold:
        raise ValueError("annotation and expected output must describe the same gold")
    inp = case["input"]
    if inp["source_item_ids"] != gold.source_item_ids or set(inp["product_contexts"]) != set(gold.source_item_ids):
        raise ValueError("case input, snapshot and gold product IDs must agree")
    if not inp["target"].strip() or any(not isinstance(v, str) or not v.strip() for v in inp["product_contexts"].values()):
        raise ValueError("case target and product contexts must be nonempty strings")
    for key, value in (("input", inp), ("expected_output", case["expected_output"]), ("annotation", case["gold_annotation"])):
        if case["metadata"][f"{key}_sha256"] != digest(value):
            raise ValueError(f"case {key} hash changed; prepare a new reviewed case version")
    return gold


def load_case(path):
    case = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_case(case)
    return case
