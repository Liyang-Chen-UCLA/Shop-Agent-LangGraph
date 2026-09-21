from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from .models import RouteResult
from .taxonomy import TAXONOMY_TOOLS


PROMPT_PATH = Path(__file__).with_name("prompts") / "route-agent.md"


def _deepseek_model() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )


class RouteAgent:
    """Public adapter whose invoke methods return RouteResult instead of graph state."""

    def __init__(self, graph: CompiledStateGraph[Any, Any, Any, Any]) -> None:
        self.graph = graph

    @staticmethod
    def _input(product: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(product, str):
            normalized = product.strip()
            if not normalized:
                raise ValueError("product must be a non-empty string")
            return {"messages": [{"role": "user", "content": normalized}]}
        if "messages" in product:
            return product
        raw_product = product.get("product")
        if not isinstance(raw_product, str) or not raw_product.strip():
            raise ValueError("input must be a product string, {'product': str}, or agent messages")
        return {"messages": [{"role": "user", "content": raw_product.strip()}]}

    @staticmethod
    def _result(state: dict[str, Any]) -> RouteResult:
        result = state.get("structured_response")
        if result is None:
            raise RuntimeError("route graph completed without a structured response")
        return result if isinstance(result, RouteResult) else RouteResult.model_validate(result)

    def invoke(self, product: str | dict[str, Any], config: Any = None, **kwargs: Any) -> RouteResult:
        return self._result(self.graph.invoke(self._input(product), config=config, **kwargs))

    async def ainvoke(
        self, product: str | dict[str, Any], config: Any = None, **kwargs: Any
    ) -> RouteResult:
        return self._result(await self.graph.ainvoke(self._input(product), config=config, **kwargs))


def build_route_agent(model: BaseChatModel | None = None) -> RouteAgent:
    """Build the LangGraph route agent, allowing an injected model for tests."""
    graph = create_agent(
        model=model or _deepseek_model(),
        tools=TAXONOMY_TOOLS,
        system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
        response_format=ToolStrategy(RouteResult),
        name="route_agent",
    )
    return RouteAgent(graph)


class LazyRouteAgent:
    """Delay model construction so imports work without secrets configured."""

    def __init__(self) -> None:
        self._instance: RouteAgent | None = None
        self._lock = Lock()

    def _get(self) -> RouteAgent:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = build_route_agent()
        return self._instance

    def invoke(self, product: str | dict[str, Any], config: Any = None, **kwargs: Any) -> RouteResult:
        return self._get().invoke(product, config=config, **kwargs)

    async def ainvoke(
        self, product: str | dict[str, Any], config: Any = None, **kwargs: Any
    ) -> RouteResult:
        return await self._get().ainvoke(product, config=config, **kwargs)


route_agent = LazyRouteAgent()
