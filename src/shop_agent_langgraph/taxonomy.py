from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain.tools import tool

from .models import RouteResult, TaxonomyNode


TAXONOMY_PATH = Path(__file__).with_name("data") / "google_product_taxonomy_zh-CN.jsonl"
SEPARATOR_PATTERN = re.compile(r"[\s/\\>、，,。.!！?？()（）\[\]【】_\-]+")
InternalNode = dict[str, Any]


def _normalize(value: str) -> str:
    return SEPARATOR_PATTERN.sub("", unicodedata.normalize("NFKC", value).casefold())


@lru_cache(maxsize=4)
def _load_taxonomy(
    taxonomy_path: Path = TAXONOMY_PATH,
) -> tuple[list[InternalNode], dict[str, InternalNode], dict[str, list[InternalNode]]]:
    nodes: list[InternalNode] = []
    by_id: dict[str, InternalNode] = {}
    children: dict[str, list[InternalNode]] = {}
    with taxonomy_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            raw_node = json.loads(line)
            node_id = str(raw_node["id"])
            if node_id in by_id:
                raise ValueError(f"duplicate taxonomy id {node_id} at line {line_number}")
            node: InternalNode = {
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


def _tool_node(node: InternalNode) -> dict[str, Any]:
    """Expose navigation metadata to tools; final output is narrowed by the schema."""
    return {
        "node_id": node["node_id"],
        "node_name": node["node_name"],
        "node_path": node["node_path"],
        "parent_id": node["parent_id"],
        "level": node["level"],
    }


def _score(query: str, node: InternalNode) -> tuple[int, int, int, str] | None:
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


@tool
def taxonomy_search_nodes(queries: list[str], limit: int = 5) -> dict[str, Any]:
    """Search taxonomy nodes for multiple product queries and return ranked matches."""
    if not queries or any(not query.strip() for query in queries):
        raise ValueError("queries must be a non-empty list of non-empty strings")
    bounded_limit = max(1, min(limit, 10))
    nodes, _, _ = _load_taxonomy()
    results: list[dict[str, Any]] = []
    for raw_query in queries:
        query = raw_query.strip()
        scored = [(_score(query, node), node) for node in nodes]
        matches = [
            node
            for score, node in sorted(
                (item for item in scored if item[0] is not None), key=lambda item: item[0]
            )
        ]
        results.append(
            {"query": query, "matches": [_tool_node(node) for node in matches[:bounded_limit]]}
        )
    return {"results": results}


@tool
def taxonomy_get_nodes(node_ids: list[str]) -> dict[str, Any]:
    """Read multiple canonical taxonomy nodes by exact node ID."""
    if not node_ids or any(not node_id.strip() for node_id in node_ids):
        raise ValueError("node_ids must be a non-empty list of non-empty strings")
    normalized_ids = list(dict.fromkeys(node_id.strip() for node_id in node_ids))
    _, by_id, _ = _load_taxonomy()
    return {
        "nodes": [_tool_node(by_id[node_id]) for node_id in normalized_ids if node_id in by_id],
        "missing_node_ids": [node_id for node_id in normalized_ids if node_id not in by_id],
    }


@tool
def taxonomy_get_children(node_ids: list[str]) -> dict[str, Any]:
    """Read all direct children of multiple taxonomy nodes."""
    if not node_ids or any(not node_id.strip() for node_id in node_ids):
        raise ValueError("node_ids must be a non-empty list of non-empty strings")
    normalized_ids = list(dict.fromkeys(node_id.strip() for node_id in node_ids))
    _, _, children = _load_taxonomy()
    return {
        "results": [
            {
                "node_id": node_id,
                "children": [_tool_node(node) for node in children.get(node_id, [])],
            }
            for node_id in normalized_ids
        ]
    }


@tool(args_schema=RouteResult)
def submit_result(
    product: str,
    status: Literal["resolved", "ambiguous"],
    resolved_nodes: list[TaxonomyNode],
    candidates: list[TaxonomyNode],
    children: list[TaxonomyNode],
) -> dict[str, Any]:
    """Submit the final taxonomy route result for runtime validation."""
    return RouteResult(
        product=product,
        status=status,
        resolved_nodes=resolved_nodes,
        candidates=candidates,
        children=children,
    ).model_dump()


TAXONOMY_TOOLS = [
    taxonomy_search_nodes,
    taxonomy_get_nodes,
    taxonomy_get_children,
    submit_result,
]
