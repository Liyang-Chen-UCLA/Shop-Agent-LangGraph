"""LangGraph-based shopping agents."""

from .agents.eval import (
    EvalAgent,
    EvalGroup,
    EvalReport,
    RelationAgent,
    build_eval_agent,
    build_relation_agent,
    eval_agent,
    relation_agent,
)
from .agents.intent import IntentResult, build_intent_agent, intent_agent
from .agents.market import MarketResult, build_market_agent, market_agent
from .agents.research import (
    PersonalizeAgent,
    ResearchResult,
    build_personalize_agent,
    build_research_agent,
    personalize_agent,
    research_agent,
)
from .agents.route import RouteResult, TaxonomyNode, build_route_agent, route_agent
from .core.config import CONFIG
from .domain.criteria import CriteriaAttributeSet
from .supervisor import (
    SupervisorState,
    build_supervisor_agent,
    build_supervisor_graph,
    supervisor,
)

__all__ = [
    "CriteriaAttributeSet",
    "CONFIG",
    "EvalAgent",
    "EvalGroup",
    "EvalReport",
    "IntentResult",
    "MarketResult",
    "PersonalizeAgent",
    "RelationAgent",
    "ResearchResult",
    "RouteResult",
    "SupervisorState",
    "TaxonomyNode",
    "build_eval_agent",
    "build_intent_agent",
    "build_market_agent",
    "build_personalize_agent",
    "build_relation_agent",
    "build_research_agent",
    "build_route_agent",
    "build_supervisor_agent",
    "build_supervisor_graph",
    "eval_agent",
    "intent_agent",
    "market_agent",
    "personalize_agent",
    "relation_agent",
    "research_agent",
    "route_agent",
    "supervisor",
]


def main() -> None:
    """Start the interactive Supervisor CLI."""
    from .cli import main as cli_main

    cli_main()
