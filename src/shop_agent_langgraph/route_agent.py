from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from .models import RouteResult
from .taxonomy import TAXONOMY_TOOLS


PROMPT_PATH = Path(__file__).with_name("prompts") / "route-agent.md"


def _deepseek_model() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )


class RouteState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route_result: NotRequired[RouteResult]


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
        result = state.get("route_result")
        if result is None:
            raise RuntimeError("route graph completed without a validated submit_result call")
        return result if isinstance(result, RouteResult) else RouteResult.model_validate(result)

    def invoke(self, product: str | dict[str, Any], config: Any = None, **kwargs: Any) -> RouteResult:
        return self._result(self.graph.invoke(self._input(product), config=config, **kwargs))

    async def ainvoke(
        self, product: str | dict[str, Any], config: Any = None, **kwargs: Any
    ) -> RouteResult:
        return self._result(await self.graph.ainvoke(self._input(product), config=config, **kwargs))


def build_route_agent(model: BaseChatModel | None = None) -> RouteAgent:
    """Build the LangGraph route agent, allowing an injected model for tests."""
    chat_model = (model or _deepseek_model()).bind_tools(TAXONOMY_TOOLS)
    system_message = SystemMessage(PROMPT_PATH.read_text(encoding="utf-8"))
    tools_by_name = {tool.name: tool for tool in TAXONOMY_TOOLS}

    def call_model(state: RouteState) -> dict[str, list[AnyMessage]]:
        response = chat_model.invoke([system_message, *state["messages"]])
        return {"messages": [response]}

    def call_tools(state: RouteState) -> dict[str, Any]:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            raise RuntimeError("tools node requires an AIMessage")

        tool_messages: list[ToolMessage] = []
        validated_result: RouteResult | None = None
        for tool_call in message.tool_calls:
            name = tool_call["name"]
            tool_call_id = tool_call["id"]
            arguments = tool_call["args"]

            if name == "submit_result":
                try:
                    validated_result = RouteResult.model_validate(arguments)
                    content = "Result accepted by runtime."
                except ValidationError as exc:
                    content = (
                        "submit_result was rejected by Pydantic validation. Fix every error and "
                        f"call submit_result again.\n{exc}"
                    )
                tool_messages.append(
                    ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
                )
                continue

            selected_tool = tools_by_name.get(name)
            if selected_tool is None:
                content = f"Unknown tool {name!r}. Use only the provided tools."
            else:
                try:
                    output = selected_tool.invoke(arguments)
                    content = (
                        output
                        if isinstance(output, str)
                        else json.dumps(output, ensure_ascii=False)
                    )
                except Exception as exc:
                    content = f"Tool {name!r} failed: {exc}. Correct the arguments and retry."
            tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))

        update: dict[str, Any] = {"messages": tool_messages}
        if validated_result is not None:
            update["route_result"] = validated_result
        return update

    def reject_plain_response(_state: RouteState) -> dict[str, list[AnyMessage]]:
        return {
            "messages": [
                HumanMessage(
                    "Runtime constraint: you may only finish by calling submit_result. "
                    "Continue using the tools, then call submit_result with the complete result."
                )
            ]
        }

    def after_model(state: RouteState) -> Literal["tools", "retry"]:
        message = state["messages"][-1]
        return "tools" if isinstance(message, AIMessage) and message.tool_calls else "retry"

    def after_tools(state: RouteState) -> Literal["end", "model"]:
        return "end" if state.get("route_result") is not None else "model"

    builder = StateGraph(RouteState)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_node("retry", reject_plain_response)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", after_model, {"tools": "tools", "retry": "retry"})
    builder.add_edge("retry", "model")
    builder.add_conditional_edges("tools", after_tools, {"end": END, "model": "model"})
    return RouteAgent(builder.compile(name="route_agent"))


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
