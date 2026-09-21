"""LangGraph-based shopping agents."""

from .intent_agent import build_intent_agent, intent_agent
from .models import IntentResult, RouteResult, TaxonomyNode
from .route_agent import build_route_agent, route_agent
from .supervisor import SupervisorState, build_supervisor_graph, supervisor

__all__ = [
    "IntentResult",
    "RouteResult",
    "SupervisorState",
    "TaxonomyNode",
    "build_intent_agent",
    "build_route_agent",
    "build_supervisor_graph",
    "intent_agent",
    "route_agent",
    "supervisor",
]


def main() -> None:
    """Start the interactive Supervisor CLI."""
    from .cli import main as cli_main

    cli_main()
