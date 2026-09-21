from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any


TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "google_product_taxonomy_zh-CN.jsonl"
)
SEPARATOR_PATTERN = re.compile(r"[\s/\\>、，,。.!！?？()（）\[\]【】_\-]+")
TaxonomyRecord = dict[str, Any]


def _normalize(value: str) -> str:
    return SEPARATOR_PATTERN.sub("", unicodedata.normalize("NFKC", value).casefold())


@lru_cache(maxsize=1)
def load_taxonomy() -> tuple[
    list[TaxonomyRecord],
    dict[str, TaxonomyRecord],
    dict[str, list[TaxonomyRecord]],
]:
    nodes: list[TaxonomyRecord] = []
    by_id: dict[str, TaxonomyRecord] = {}
    children: dict[str, list[TaxonomyRecord]] = {}
    with TAXONOMY_PATH.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            raw_node = json.loads(line)
            node_id = str(raw_node["id"])
            if node_id in by_id:
                raise ValueError(f"duplicate taxonomy id {node_id} at line {line_number}")
            node: TaxonomyRecord = {
                "node_id": node_id,
                "node_name": str(raw_node["name"]),
                "node_path": str(raw_node["path"]),
                "parent_id": None
                if raw_node.get("parent_id") is None
                else str(raw_node["parent_id"]),
                "level": int(raw_node["level"]),
            }
            nodes.append(node)
            by_id[node_id] = node
            if node["parent_id"] is not None:
                children.setdefault(node["parent_id"], []).append(node)
    for child_nodes in children.values():
        child_nodes.sort(key=lambda item: (item["node_name"], item["node_id"]))
    return nodes, by_id, children


def _score(query: str, node: TaxonomyRecord) -> tuple[int, int, int, str] | None:
    normalized_query = _normalize(query)
    if not normalized_query:
        return None
    normalized_name = _normalize(node["node_name"])
    normalized_path = _normalize(node["node_path"])
    if normalized_query == normalized_name:
        rank = 0
    elif normalized_name in normalized_query:
        rank = 1
    elif normalized_query in normalized_name:
        rank = 2
    elif normalized_query in normalized_path:
        rank = 3
    else:
        query_chars = set(normalized_query)
        overlap = len(query_chars.intersection(normalized_name))
        minimum_overlap = max(2, min(len(query_chars), len(set(normalized_name))) // 2)
        if overlap < minimum_overlap:
            return None
        rank = 4
    return rank, abs(len(normalized_name) - len(normalized_query)), -node["level"], node["node_id"]


def search_nodes(queries: list[str], limit: int) -> list[dict[str, Any]]:
    nodes, _, _ = load_taxonomy()
    results: list[dict[str, Any]] = []
    for query in queries:
        scored = [(_score(query, node), node) for node in nodes]
        matches = [
            node
            for score, node in sorted(
                (item for item in scored if item[0] is not None), key=lambda item: item[0]
            )
        ]
        results.append({"query": query, "matches": matches[:limit]})
    return results


def get_nodes(node_ids: list[str]) -> tuple[list[TaxonomyRecord], list[str]]:
    _, by_id, _ = load_taxonomy()
    found = [by_id[node_id] for node_id in node_ids if node_id in by_id]
    missing = [node_id for node_id in node_ids if node_id not in by_id]
    return found, missing


def get_children(node_ids: list[str]) -> list[dict[str, Any]]:
    _, _, children = load_taxonomy()
    return [
        {"node_id": node_id, "children": children.get(node_id, [])}
        for node_id in node_ids
    ]
