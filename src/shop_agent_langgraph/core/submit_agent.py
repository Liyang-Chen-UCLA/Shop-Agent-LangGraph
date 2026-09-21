from __future__ import annotations

import json
from typing import Annotated, Any, Literal, NotRequired, Sequence, TypedDict, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ValidationError


ResultT = TypeVar("ResultT", bound=BaseModel)


class SubmitAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    submitted_result: NotRequired[BaseModel]


def build_submit_agent_graph(
    *,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    result_schema: type[ResultT],
    submit_tool_name: str,
    name: str,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build a tool-loop graph that ends only after a valid submission call."""
    chat_model = model.bind_tools(tools)
    system_message = SystemMessage(system_prompt)
    tools_by_name = {tool.name: tool for tool in tools}

    def call_model(state: SubmitAgentState) -> dict[str, list[AnyMessage]]:
        response = chat_model.invoke([system_message, *state["messages"]])
        return {"messages": [response]}

    def call_tools(state: SubmitAgentState) -> dict[str, Any]:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            raise RuntimeError("tools node requires an AIMessage")

        tool_messages: list[ToolMessage] = []
        submitted_result: ResultT | None = None
        for tool_call in message.tool_calls:
            tool_name = tool_call["name"]
            tool_call_id = tool_call["id"]
            arguments = tool_call["args"]

            if tool_name == submit_tool_name:
                try:
                    submitted_result = result_schema.model_validate(arguments)
                    content = "Result accepted by runtime."
                except ValidationError as exc:
                    content = (
                        f"{submit_tool_name} was rejected by Pydantic validation. "
                        f"Fix every error and call {submit_tool_name} again.\n{exc}"
                    )
                tool_messages.append(
                    ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)
                )
                continue

            selected_tool = tools_by_name.get(tool_name)
            if selected_tool is None:
                content = f"Unknown tool {tool_name!r}. Use only the provided tools."
            else:
                try:
                    output = selected_tool.invoke(arguments)
                    content = (
                        output
                        if isinstance(output, str)
                        else json.dumps(output, ensure_ascii=False)
                    )
                except Exception as exc:
                    content = (
                        f"Tool {tool_name!r} failed: {exc}. Correct the arguments and retry."
                    )
            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)
            )

        update: dict[str, Any] = {"messages": tool_messages}
        if submitted_result is not None:
            update["submitted_result"] = submitted_result
        return update

    def reject_plain_response(_state: SubmitAgentState) -> dict[str, list[AnyMessage]]:
        return {
            "messages": [
                HumanMessage(
                    f"Runtime constraint: you may only finish by calling {submit_tool_name}. "
                    f"Continue working, then call {submit_tool_name} with the complete result."
                )
            ]
        }

    def after_model(state: SubmitAgentState) -> Literal["tools", "retry"]:
        message = state["messages"][-1]
        return "tools" if isinstance(message, AIMessage) and message.tool_calls else "retry"

    def after_tools(state: SubmitAgentState) -> Literal["end", "model"]:
        return "end" if state.get("submitted_result") is not None else "model"

    builder = StateGraph(SubmitAgentState)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_node("retry", reject_plain_response)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", after_model, {"tools": "tools", "retry": "retry"})
    builder.add_edge("retry", "model")
    builder.add_conditional_edges("tools", after_tools, {"end": END, "model": "model"})
    return builder.compile(name=name)
