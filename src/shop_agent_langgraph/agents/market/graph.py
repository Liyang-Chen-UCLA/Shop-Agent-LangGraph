from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel

from ...core.config import CONFIG
from ...core.llm import build_deepseek_model
from ...domain.market_env import get_product_pages
from ..research.graph import research_agent
from ..research.schemas import ResearchResult
from .aggregation import MarketAggregationAgent
from .schemas import MarketResult, MarketSelection
from .screening import screening_session
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

    async def ainvoke_fixed_products(self, item_ids: list[str], target: str, *,
                                    product_contexts: dict[str, str], research,
                                    config: dict | None = None, concurrency: int = 3) -> MarketResult:
        """Evaluate delivery on an explicit snapshot without search or resampling."""
        if not item_ids or len(item_ids) != len(set(item_ids)):
            raise ValueError("fixed products require nonempty distinct IDs")
        if set(product_contexts) != set(item_ids) or not target.strip() or concurrency < 1:
            raise ValueError("fixed snapshot must cover every product and have a target and positive concurrency")
        semaphore = asyncio.Semaphore(concurrency)

        async def read(item_id):
            async with semaphore:
                result = await research.ainvoke(item_id, target, raw_text=product_contexts[item_id], config=config)
                if result.item_id != item_id or result.relevance != "relevant":
                    raise ValueError(f"fixed product research did not confirm the requested product: {item_id}")
                return result.model_copy(update={"evidence": [e for e in result.evidence
                                           if e.subject in {"product", "variant"}]})

        results = await asyncio.gather(*(read(item_id) for item_id in item_ids))
        merged = await self.aggregator.ainvoke(results, target, config=config)
        return MarketResult(query=target, item_ids=list(item_ids), status="completed",
                            criteria=merged.criteria, attributes=merged.attributes,
                            audit=[{"stage": "fixed_research", **r.model_dump(mode="json")} for r in results])

    async def _select(self, query: str, target: str = "") -> MarketSelection:
        with screening_session() as session:
            state = await self.selection_graph.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Product query: {query}\n"
                                f"Target category: {target or query}\n"
                                f"Maximum research sample: {CONFIG.market.max_search_products}\n"
                                f"Maximum search candidates: {CONFIG.market.max_search_candidates}\n"
                                f"Page reading calls per product: {CONFIG.market.max_page_reads_per_product}\n"
                                f"Pages per call: {CONFIG.market.max_pages_per_read}\n"
                                "Full-context fallback: once per product after focused reading.\n"
                                "Screen only against Target category, not personal preferences."
                            ),
                        }
                    ]
                },
                config={"recursion_limit": 2 * CONFIG.market.max_search_candidates * (
                    CONFIG.market.max_page_reads_per_product + 3) + 10},
            )
            raw = state["structured_response"]
            selection = raw if isinstance(raw, MarketSelection) else MarketSelection.model_validate(raw)
            if len(selection.item_ids) != len(set(selection.item_ids)):
                raise ValueError("research sample must contain distinct product IDs")
            decisions = {s.item_id: s for s in selection.screenings}
            if (len(decisions) != len(selection.screenings)
                    or set(decisions) != session.inspected):
                raise ValueError("screening must decide every inspected product exactly once")
            for decision in selection.screenings:
                session.require_candidate(decision.item_id)
                if not decision.reason.strip():
                    raise ValueError("screening reason must be nonblank")
                pages = session.pages.get(decision.item_id, {})
                if (decision.relevance == "uncertain"
                        and decision.item_id not in session.full_reads
                        and len(pages) < len(get_product_pages(decision.item_id))):
                    raise ValueError("uncertain products require full-context escalation or all pages read")
                for evidence in decision.evidence:
                    page = pages.get(evidence.page_id)
                    if (page is None or not evidence.source_text.strip()
                            or evidence.source_text not in page["text"]):
                        raise ValueError("screening evidence must quote an actually read OCR page")
            for item_id in selection.item_ids:
                if item_id not in decisions or decisions[item_id].relevance != "relevant":
                    raise ValueError("selected products require confirmed node relevance")
            selection._read_context = {item_id: dict(
                pages=list(session.pages.get(item_id, {}).values()),
                page_read_calls=session.page_reads.get(item_id, 0),
                full_context_read=item_id in session.full_reads,
            ) for item_id in session.candidates}
            return selection

    async def _research(self, item_ids: list[str], target: str = "", *,
                        pre_read: dict | None = None) -> list[ResearchResult]:
        results = await asyncio.gather(
            *(research_agent.ainvoke(item_id, target, pre_read=pre_read[item_id])
              if pre_read else research_agent.ainvoke(item_id, target) for item_id in item_ids)
        )
        return list(results)

    async def ainvoke(self, query: str, target: str = "") -> MarketResult:
        selection = await self._select(query, target)
        if selection._read_context:
            decisions = {s.item_id: s.model_dump() for s in selection.screenings}
            pre_read = {item_id: {"screening": decisions[item_id],
                                 **selection._read_context[item_id]}
                        for item_id in selection.item_ids}
            researched = await self._research(selection.item_ids, target or query, pre_read=pre_read)
        else:
            researched = await self._research(selection.item_ids, target or query)
        accepted = [r.model_copy(update={"evidence": [e for e in r.evidence
                    if e.subject in {"product", "variant"}]})
                    for r in researched if r.relevance == "relevant"]
        audit = [{"stage": "research", "item_id": r.item_id, "relevance": r.relevance,
                  "reason": r.relevance_reason} for r in researched]
        audit.extend({"stage": "screening", **s.model_dump(),
                      "selected": s.item_id in selection.item_ids,
                      "read_page_ids": [p["page_id"] for p in selection._read_context[s.item_id]["pages"]],
                      "page_read_calls": selection._read_context[s.item_id]["page_read_calls"],
                      "full_context_read": selection._read_context[s.item_id]["full_context_read"]}
                     for s in selection.screenings)
        screened = {s.item_id for s in selection.screenings}
        audit.extend({"stage": "screening", "item_id": item_id, "relevance": "not_assessed",
                      "selected": False, "reason": "Not inspected for the bounded sample"}
                     for item_id in sorted(selection._read_context.keys() - screened))
        if not accepted:
            from .schemas import PendingAggregation
            return MarketResult(query=query, item_ids=[], status="pending", criteria=[],
                attributes=[], audit=audit, pending_groups=[PendingAggregation(
                    group_id="category_scope", left_source_item_ids=selection.item_ids,
                    right_source_item_ids=[], left_ids=[], right_ids=[],
                    reason="No confirmed in-scope products", missing_evidence=["Relevant product evidence"])])
        merged = await self.aggregator.ainvoke(accepted, target or query)
        return MarketResult(
            query=query,
            item_ids=[r.item_id for r in accepted],
            audit=audit,
            status="completed",
            criteria=merged.criteria,
            attributes=merged.attributes,
        )

    def invoke(self, query: str, target: str = "") -> MarketResult:
        return asyncio.run(self.ainvoke(query, target))


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

    def invoke(self, query: str, target: str = "") -> MarketResult:
        return self._get().invoke(query, target)

    async def ainvoke(self, query: str, target: str = "") -> MarketResult:
        return await self._get().ainvoke(query, target)


market_agent = LazyMarketAgent()
