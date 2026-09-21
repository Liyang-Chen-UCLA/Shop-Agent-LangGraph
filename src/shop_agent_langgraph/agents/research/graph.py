from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from ...core.llm import build_deepseek_model
from ...core.submit_agent import build_submit_agent_graph
from ...domain.market_env import get_product_raw_text
from .schemas import ResearchResult
from .tools import build_research_tools


PROMPT_PATH = Path(__file__).with_name("prompt.md")


class ResearchAgent:
    def __init__(self, graph: CompiledStateGraph[Any, Any, Any, Any]) -> None:
        self.graph = graph

    @staticmethod
    def _input(item_id: str) -> dict[str, Any]:
        raw_text = get_product_raw_text(item_id)
        content = (
            f"Analyze market item `{item_id}`.\n"
            f"<product_context item_id=\"{item_id}\">\n{raw_text}\n</product_context>"
        )
        return {"messages": [HumanMessage(content)]}

    @staticmethod
    def _result(state: dict[str, Any]) -> ResearchResult:
        result = state["submitted_result"]
        return result if isinstance(result, ResearchResult) else ResearchResult.model_validate(result)

    def invoke(self, item_id: str) -> ResearchResult:
        return self._result(self.graph.invoke(self._input(str(item_id))))

    async def ainvoke(self, item_id: str) -> ResearchResult:
        return self._result(await self.graph.ainvoke(self._input(str(item_id))))


def build_research_agent(model: BaseChatModel | None = None) -> ResearchAgent:
    graph = build_submit_agent_graph(
        model=model or build_deepseek_model(),
        tools=build_research_tools(),
        system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
        result_schema=ResearchResult,
        submit_tool_name="submit_research_result",
        name="research_agent",
    )
    return ResearchAgent(graph)


class LazyResearchAgent:
    def __init__(self) -> None:
        self._instance: ResearchAgent | None = None
        self._lock = Lock()

    def _get(self) -> ResearchAgent:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = build_research_agent()
        return self._instance

    def invoke(self, item_id: str) -> ResearchResult:
        return self._get().invoke(item_id)

    async def ainvoke(self, item_id: str) -> ResearchResult:
        return await self._get().ainvoke(item_id)


research_agent = LazyResearchAgent()
