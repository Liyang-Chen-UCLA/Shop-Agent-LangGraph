from .graph import EvalAgent, build_eval_agent, eval_agent
from .schemas import EvalGroup, EvalReport

RelationAgent = EvalAgent
build_relation_agent = build_eval_agent
relation_agent = eval_agent

__all__ = [
    "EvalAgent",
    "EvalGroup",
    "EvalReport",
    "RelationAgent",
    "build_eval_agent",
    "build_relation_agent",
    "eval_agent",
    "relation_agent",
]
