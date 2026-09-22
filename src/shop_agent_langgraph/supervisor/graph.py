from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..core.llm import build_deepseek_model
from .state import SupervisorState
from .tools import SUPERVISOR_TOOLS


PROMPT_PATH = Path(__file__).with_name("prompt.md")


def build_supervisor_agent(
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    return create_agent(
        model=model or build_deepseek_model(),
        tools=SUPERVISOR_TOOLS,
        system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
        name="supervisor_agent",
    )


def build_supervisor_graph(
    model: BaseChatModel | None = None,
    *,
    checkpointer: Any = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build the messages-only entry graph: START -> supervisor -> END."""
    supervisor_agent = build_supervisor_agent(model)

    def run_supervisor(state: SupervisorState) -> dict[str, list[Any]]:
        result = supervisor_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][len(state["messages"]):]}

    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", run_supervisor)
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", END)
    return builder.compile(checkpointer=checkpointer, name="shop_agent")


class Supervisor:
    """User-facing conversational entry point backed by the outer graph."""

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
