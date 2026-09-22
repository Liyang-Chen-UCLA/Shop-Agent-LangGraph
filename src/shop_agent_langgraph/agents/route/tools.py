from __future__ import annotations

from typing import Any, Literal

from langchain.tools import tool

from ...domain.taxonomy import get_children, get_nodes, search_nodes
from .schemas import RouteResult, TaxonomyNode


@tool
def taxonomy_search_nodes(queries: list[str], limit: int = 5) -> dict[str, Any]:
    """Search taxonomy nodes for multiple product queries and return ranked matches."""
    if not queries or any(not query.strip() for query in queries):
        raise ValueError("queries must be a non-empty list of non-empty strings")
    normalized = [query.strip() for query in queries]
    return {"results": search_nodes(normalized, max(1, min(limit, 10)))}


@tool
def taxonomy_get_nodes(node_ids: list[str]) -> dict[str, Any]:
    """Read multiple canonical taxonomy nodes by exact node ID."""
    if not node_ids or any(not node_id.strip() for node_id in node_ids):
        raise ValueError("node_ids must be a non-empty list of non-empty strings")
    normalized = list(dict.fromkeys(node_id.strip() for node_id in node_ids))
    nodes, missing = get_nodes(normalized)
    return {"nodes": nodes, "missing_node_ids": missing}


@tool
def taxonomy_get_children(node_ids: list[str]) -> dict[str, Any]:
    """Read all direct children of multiple taxonomy nodes."""
    if not node_ids or any(not node_id.strip() for node_id in node_ids):
        raise ValueError("node_ids must be a non-empty list of non-empty strings")
    normalized = list(dict.fromkeys(node_id.strip() for node_id in node_ids))
    return {"results": get_children(normalized)}


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

ROUTE_TOOLS = [
    taxonomy_search_nodes,
    taxonomy_get_nodes,
    taxonomy_get_children,
]
