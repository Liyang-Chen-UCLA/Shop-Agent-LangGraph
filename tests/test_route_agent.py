from typing import Any

from shop_agent_langgraph.models import RouteResult
from shop_agent_langgraph.route_agent import RouteAgent


class StubGraph:
    def invoke(self, state: dict[str, Any], config: Any = None, **kwargs: Any) -> dict[str, Any]:
        product = state["messages"][0]["content"]
        return {
            "route_result": {
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
