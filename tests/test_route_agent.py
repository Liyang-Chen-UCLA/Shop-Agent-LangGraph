from typing import Any

from shop_agent_langgraph.agents.route.graph import RouteAgent
from shop_agent_langgraph.agents.route.schemas import RouteResult


class StubGraph:
    def invoke(self, state: dict[str, Any], config: Any = None, **kwargs: Any) -> dict[str, Any]:
        product = state["messages"][0]["content"]
        return {
            "submitted_result": {
                "product": product,
                "status": "ambiguous",
                "resolved_nodes": [],
                "candidates": [],
                "children": [],
            }
        }


def test_invoke_returns_route_result() -> None:
    agent = RouteAgent(StubGraph())  # type: ignore[arg-type]
    result = agent.invoke({"product": "机械键盘"})
    assert isinstance(result, RouteResult)
    assert result.product == "机械键盘"
