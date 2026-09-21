"""LangGraph-based shopping agents."""

from .intent_agent import build_intent_agent, intent_agent
from .models import IntentResult, RouteResult, TaxonomyNode
from .route_agent import build_route_agent, route_agent

__all__ = [
    "IntentResult",
    "RouteResult",
    "TaxonomyNode",
    "build_intent_agent",
    "build_route_agent",
    "intent_agent",
    "route_agent",
]


def main() -> None:
    """Keep the package script useful without starting an interactive runtime."""
    print("Import `intent_agent` or `route_agent` and call `.invoke(...)`.")
