from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from ...core.llm import build_deepseek_model
from .schemas import IntentResult


PROMPT_PATH = Path(__file__).with_name("prompt.md")


class IntentAgent:
    """Analyze user shopping requests and return an IntentResult."""

    def __init__(self, graph: CompiledStateGraph[Any, Any, Any, Any]) -> None:
        self.graph = graph

    @staticmethod
    def _input(request: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(request, str):
            content = request.strip()
            if not content:
                raise ValueError("request must be a non-empty string")
            return {"messages": [{"role": "user", "content": content}]}
        if "messages" in request:
            return request
        content = request.get("request")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("input must be a request string, {'request': str}, or agent messages")
        return {"messages": [{"role": "user", "content": content.strip()}]}

    @staticmethod
    def _result(state: dict[str, Any]) -> IntentResult:
        result = state["structured_response"]
        return result if isinstance(result, IntentResult) else IntentResult.model_validate(result)

    def invoke(self, request: str | dict[str, Any], config: Any = None, **kwargs: Any) -> IntentResult:
        return self._result(self.graph.invoke(self._input(request), config=config, **kwargs))

    async def ainvoke(
        self, request: str | dict[str, Any], config: Any = None, **kwargs: Any
    ) -> IntentResult:
        state = await self.graph.ainvoke(self._input(request), config=config, **kwargs)
        return self._result(state)


def build_intent_agent(model: BaseChatModel | None = None) -> IntentAgent:
    graph = create_agent(
        model=model or build_deepseek_model(),
        tools=[],
        system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
        response_format=ToolStrategy(IntentResult),
        name="intent_agent",
    )
    return IntentAgent(graph)


class LazyIntentAgent:
    """Delay model construction so imports work without secrets configured."""

    def __init__(self) -> None:
        self._instance: IntentAgent | None = None
        self._lock = Lock()

    def _get(self) -> IntentAgent:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = build_intent_agent()
        return self._instance

    def invoke(self, request: str | dict[str, Any], config: Any = None, **kwargs: Any) -> IntentResult:
        return self._get().invoke(request, config=config, **kwargs)

    async def ainvoke(
        self, request: str | dict[str, Any], config: Any = None, **kwargs: Any
    ) -> IntentResult:
        return await self._get().ainvoke(request, config=config, **kwargs)


intent_agent = LazyIntentAgent()
