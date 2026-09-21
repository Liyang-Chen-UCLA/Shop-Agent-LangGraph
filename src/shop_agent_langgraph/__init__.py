"""LangGraph-based shopping agents."""

from .models import RouteResult, TaxonomyNode
from .route_agent import build_route_agent, route_agent

__all__ = ["RouteResult", "TaxonomyNode", "build_route_agent", "route_agent"]


def main() -> None:
    """Keep the package script useful without starting an interactive runtime."""
    print("Import `route_agent` and call `route_agent.invoke(product)`.")
