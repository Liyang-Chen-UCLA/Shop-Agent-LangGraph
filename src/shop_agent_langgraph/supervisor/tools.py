from __future__ import annotations

from langchain.tools import tool

from ..agents.intent.graph import intent_agent
from ..agents.market.graph import market_agent
from ..agents.route.graph import route_agent
from ..agents.relation.graph import relation_agent
from ..domain.market_mapping import dataset_category_for_node
from ..domain.market_cache import MarketCache


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
    """Reuse saved criteria and attributes for a resolved node, researching only on a cache miss."""
    cache = MarketCache()
    node = cache.node(node_id)
    cached = cache.load(node)
    if cached is not None:
        return cached.model_dump_json()
    query = dataset_category_for_node(node_id)
    result = market_agent.invoke(query)
    if result.status == "completed":
        cache.save(node, result)
    return result.model_dump_json()


@tool
def call_relation_agent(node_id: str) -> str:
    """Reuse or research a relation graph after this node's completed Market Profile is saved."""
    return relation_agent.invoke(node_id).model_dump_json()


SUPERVISOR_TOOLS = [
    call_intent_agent,
    call_route_agent,
    call_market_agent,
    call_relation_agent,
]
