from shop_agent_langgraph.agents.route.tools import (
    taxonomy_get_children,
    taxonomy_get_nodes,
    taxonomy_search_nodes,
)


def test_search_exact_node_name_first() -> None:
    payload = taxonomy_search_nodes.invoke({"queries": ["五金泵"], "limit": 3})
    assert payload["results"][0]["matches"][0]["node_id"] == "500096"


def test_get_nodes_reports_missing_ids_and_deduplicates() -> None:
    payload = taxonomy_get_nodes.invoke({"node_ids": ["632", "missing", "632"]})
    assert [node["node_id"] for node in payload["nodes"]] == ["632"]
    assert payload["missing_node_ids"] == ["missing"]


def test_get_children_returns_direct_children_in_stable_order() -> None:
    payload = taxonomy_get_children.invoke({"node_ids": ["500096"]})
    children = payload["results"][0]["children"]
    assert children
    assert all(node["parent_id"] == "500096" for node in children)
    assert [node["node_name"] for node in children] == sorted(
        node["node_name"] for node in children
    )
