from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph

from .intent_agent import intent_agent
from .llm import build_deepseek_model
from .models import IntentResult, RouteResult
from .route_agent import route_agent


PROMPT_PATH = Path(__file__).with_name("prompts") / "supervisor.md"


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    # 本轮解析结果
    intent: IntentResult | None

    # taxonomy 路由状态
    route: RouteResult | None


def build_supervisor_graph(
    model: BaseChatModel | None = None,
    *,
    checkpointer: Any = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build the top-level conversational graph."""
    chat_model = model or build_deepseek_model()
    supervisor_prompt = SystemMessage(PROMPT_PATH.read_text(encoding="utf-8"))

    def analyze_intent(state: SupervisorState) -> dict[str, Any]:
        parsed = intent_agent.invoke({"messages": state["messages"]})
        update: dict[str, Any] = {"intent": parsed}
        if parsed.action == "create":
            update["route"] = None
        return update

    def should_route(state: SupervisorState) -> Literal["route", "respond"]:
        parsed = state["intent"]
        if parsed and parsed.action in {"create", "switch"} and parsed.category:
            return "route"
        return "respond"

    def resolve_route(state: SupervisorState) -> dict[str, RouteResult]:
        parsed = state["intent"]
        if parsed is None or parsed.category is None:
            raise RuntimeError("route node requires an intent category")
        return {"route": route_agent.invoke(parsed.category)}

    def respond(state: SupervisorState) -> dict[str, list[AnyMessage]]:
        intent_context = (
            state["intent"].model_dump_json(indent=2) if state.get("intent") else "null"
        )
        route_context = (
            state["route"].model_dump_json(indent=2) if state.get("route") else "null"
        )
        runtime_context = SystemMessage(
            "Current runtime state for this turn:\n"
            f"intent = {intent_context}\n"
            f"route = {route_context}\n"
            "Answer the user's latest original message directly."
        )
        response = chat_model.invoke([supervisor_prompt, runtime_context, *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(SupervisorState)
    builder.add_node("intent_agent", analyze_intent)
    builder.add_node("route_agent", resolve_route)
    builder.add_node("supervisor", respond)
    builder.add_edge(START, "intent_agent")
    builder.add_conditional_edges(
        "intent_agent",
        should_route,
        {"route": "route_agent", "respond": "supervisor"},
    )
    builder.add_edge("route_agent", "supervisor")
    builder.add_edge("supervisor", END)
    return builder.compile(checkpointer=checkpointer, name="supervisor")


class Supervisor:
    """User-facing conversational entry point backed by a persistent graph."""

    def __init__(self, graph: CompiledStateGraph[Any, Any, Any, Any]) -> None:
        self.graph = graph

    @staticmethod
    def _config(thread_id: str, config: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(config or {})
        configurable = dict(merged.get("configurable", {}))
        configurable.setdefault("thread_id", thread_id)
        merged["configurable"] = configurable
        return merged

    def invoke(
        self,
        user_input: str,
        *,
        thread_id: str = "default",
        config: dict[str, Any] | None = None,
    ) -> str:
        content = user_input.strip()
        if not content:
            raise ValueError("user_input must be a non-empty string")
        state = self.graph.invoke(
            {"messages": [{"role": "user", "content": content}]},
            config=self._config(thread_id, config),
        )
        response = state["messages"][-1]
        if not isinstance(response, AIMessage):
            raise RuntimeError("supervisor completed without an AI response")
        return str(response.content)

    async def ainvoke(
        self,
        user_input: str,
        *,
        thread_id: str = "default",
        config: dict[str, Any] | None = None,
    ) -> str:
        content = user_input.strip()
        if not content:
            raise ValueError("user_input must be a non-empty string")
        state = await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": content}]},
            config=self._config(thread_id, config),
        )
        response = state["messages"][-1]
        if not isinstance(response, AIMessage):
            raise RuntimeError("supervisor completed without an AI response")
        return str(response.content)

    def get_state(self, thread_id: str = "default") -> SupervisorState:
        snapshot = self.graph.get_state(self._config(thread_id, None))
        return snapshot.values  # type: ignore[return-value]


class LazySupervisor:
    """Delay graph and model construction until the first conversation turn."""

    def __init__(self) -> None:
        self._instance: Supervisor | None = None
        self._lock = Lock()

    def _get(self) -> Supervisor:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = Supervisor(
                        build_supervisor_graph(checkpointer=InMemorySaver())
                    )
        return self._instance

    def invoke(
        self,
        user_input: str,
        *,
        thread_id: str = "default",
        config: dict[str, Any] | None = None,
    ) -> str:
        return self._get().invoke(user_input, thread_id=thread_id, config=config)

    async def ainvoke(
        self,
        user_input: str,
        *,
        thread_id: str = "default",
        config: dict[str, Any] | None = None,
    ) -> str:
        return await self._get().ainvoke(user_input, thread_id=thread_id, config=config)

    def get_state(self, thread_id: str = "default") -> SupervisorState:
        return self._get().get_state(thread_id)


supervisor = LazySupervisor()
