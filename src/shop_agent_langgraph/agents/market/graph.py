from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel

from ...core.config import CONFIG
from ...core.llm import build_deepseek_model
from ...domain.criteria import CriteriaAttributeSet
from ..research.graph import research_agent
from .aggregation import MarketAggregationAgent, deterministic_premerge
from .schemas import MarketResult, MarketSelection
from .tools import MARKET_TOOLS


PROMPT_PATH = Path(__file__).with_name("prompt.md")


class MarketAgent:
    def __init__(
        self,
        selection_graph: object,
        aggregator: MarketAggregationAgent,
    ) -> None:
        self.selection_graph = selection_graph
        self.aggregator = aggregator

    async def _select(self, query: str) -> MarketSelection:
        state = await self.selection_graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Product query: {query}\n"
                            f"Maximum products: {CONFIG.market.max_search_products}"
                        ),
                    }
                ]
            }
        )
        result = state["structured_response"]
        return (
            result
            if isinstance(result, MarketSelection)
            else MarketSelection.model_validate(result)
        )

    async def _research(self, item_ids: list[str]) -> list[CriteriaAttributeSet]:
        results = await asyncio.gather(
            *(research_agent.ainvoke(item_id) for item_id in item_ids)
        )
        return [
            CriteriaAttributeSet(
                source_item_ids=[result.item_id],
                criteria=result.criteria,
                attributes=result.attributes,
            )
            for result in results
        ]

    async def ainvoke(self, query: str) -> MarketResult:
        selection = await self._select(query)
        researched = await self._research(selection.item_ids)
        premerged = deterministic_premerge(researched)
        merged = await self.aggregator.ainvoke(premerged)
        return MarketResult(
            query=query,
            item_ids=selection.item_ids,
            status="completed",
            criteria=merged.criteria,
            attributes=merged.attributes,
        )

    def invoke(self, query: str) -> MarketResult:
        return asyncio.run(self.ainvoke(query))


def build_market_agent(model: BaseChatModel | None = None) -> MarketAgent:
    selected_model = model or build_deepseek_model()
    graph = create_agent(
        model=selected_model,
        tools=MARKET_TOOLS,
        system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
        response_format=ToolStrategy(MarketSelection),
        name="market_agent",
    )
    return MarketAgent(
        graph,
        aggregator=MarketAggregationAgent(selected_model),
    )


class LazyMarketAgent:
    def __init__(self) -> None:
        self._instance: MarketAgent | None = None
        self._lock = Lock()

    def _get(self) -> MarketAgent:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = build_market_agent()
        return self._instance

    def invoke(self, query: str) -> MarketResult:
        return self._get().invoke(query)

    async def ainvoke(self, query: str) -> MarketResult:
        return await self._get().ainvoke(query)


market_agent = LazyMarketAgent()
