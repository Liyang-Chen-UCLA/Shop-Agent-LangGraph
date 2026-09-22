from __future__ import annotations

from langchain.tools import tool

from ..agents.intent.graph import intent_agent
from ..agents.market.graph import market_agent
from ..agents.route.graph import route_agent
from ..domain.market_mapping import dataset_category_for_node


@tool
def call_intent_agent(request: str) -> str:
    """Interpret a shopping request, including any relevant conversation context."""
    return intent_agent.invoke(request).model_dump_json()


@tool
def call_route_agent(product: str) -> str:
    """Resolve one normalized product name to the canonical product taxonomy."""
    return route_agent.invoke(product).model_dump_json()


@tool
def call_market_agent(node_id: str) -> str:
    """Analyze the local market for a resolved taxonomy node ID."""
    query = dataset_category_for_node(node_id)
    return market_agent.invoke(query).model_dump_json()


SUPERVISOR_TOOLS = [
    call_intent_agent,
    call_route_agent,
    call_market_agent,
]
