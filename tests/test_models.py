import pytest
from pydantic import ValidationError

from shop_agent_langgraph.agents.route.schemas import RouteResult, TaxonomyNode


NODE = TaxonomyNode(node_id="1", node_name="测试", node_path="测试")


def test_resolved_result_requires_exactly_one_node() -> None:
    result = RouteResult(
        product="测试商品",
        status="resolved",
        resolved_nodes=[NODE],
        candidates=[],
        children=[],
    )
    assert result.resolved_nodes == [NODE]


def test_ambiguous_result_rejects_resolved_payload() -> None:
    with pytest.raises(ValidationError):
        RouteResult(
            product="测试商品",
            status="ambiguous",
            resolved_nodes=[NODE],
            candidates=[],
            children=[],
        )


def test_result_rejects_more_than_three_candidates() -> None:
    with pytest.raises(ValidationError):
        RouteResult(
            product="测试商品",
            status="ambiguous",
            resolved_nodes=[],
            candidates=[NODE, NODE, NODE, NODE],
            children=[],
        )
