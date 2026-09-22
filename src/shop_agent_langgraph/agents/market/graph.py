from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock

from langchain_core.language_models import BaseChatModel

from ...core.config import CONFIG
from ...core.llm import build_deepseek_model
from ...core.submit_agent import build_submit_agent_graph
from ...domain.criteria import CriteriaAttributeSet
from ..eval.graph import EvalAgent
from ..research.graph import research_agent
from .aggregation import MarketAggregationAgent
from .schemas import MarketAggregationOutcome, MarketResult, MarketSelection
from .tools import MARKET_TOOLS, submit_market_selection


PROMPT_PATH = Path(__file__).with_name("prompt.md")


class MarketAgent:
    def __init__(
        self,
        selection_graph: object,
        evaluator: EvalAgent,
        aggregator: MarketAggregationAgent,
    ) -> None:
        self.selection_graph = selection_graph
        self.evaluator = evaluator
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
        result = state["submitted_result"]
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

    async def _aggregate(
        self,
        results: list[CriteriaAttributeSet],
    ) -> MarketAggregationOutcome:
        current = results
        while len(current) > 1:
            pairs = [
                (current[index], current[index + 1])
                for index in range(0, len(current) - 1, 2)
            ]
            outcomes = await asyncio.gather(
                *(self._aggregate_pair(left, right) for left, right in pairs)
            )
            pending = [outcome for outcome in outcomes if outcome.status == "pending"]
            if pending:
                return MarketAggregationOutcome(
                    status="pending",
                    pending_groups=[
                        group for outcome in pending for group in outcome.pending_groups
                    ],
                    unresolved=[
                        reference for outcome in pending for reference in outcome.unresolved
                    ],
                )
            merged = [
                outcome.collection
                for outcome in outcomes
                if outcome.collection is not None
            ]
            current = [*merged, *([] if len(current) % 2 == 0 else [current[-1]])]
        return MarketAggregationOutcome(status="completed", collection=current[0])

    async def _aggregate_pair(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> MarketAggregationOutcome:
        report = await self.evaluator.ainvoke(left, right)
        return await self.aggregator.ainvoke(left, right, report)

    async def ainvoke(self, query: str) -> MarketResult:
        selection = await self._select(query)
        researched = await self._research(selection.item_ids)
        aggregation = await self._aggregate(researched)
        if aggregation.status == "pending":
            return MarketResult(
                query=query,
                item_ids=selection.item_ids,
                status="pending",
                criteria=[],
                attributes=[],
                pending_groups=aggregation.pending_groups,
            )
        merged = aggregation.collection
        if merged is None:
            raise RuntimeError("completed aggregation is missing its collection")
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
    graph = build_submit_agent_graph(
        model=selected_model,
        tools=[*MARKET_TOOLS, submit_market_selection],
        system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
        result_schema=MarketSelection,
        submit_tool_name="submit_market_selection",
        name="market_agent",
    )
    return MarketAgent(
        graph,
        evaluator=EvalAgent(selected_model),
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
