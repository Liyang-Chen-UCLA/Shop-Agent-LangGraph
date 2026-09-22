from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from ...domain.criteria import Attribute, CriteriaAttributeSet, Criterion


PROMPT_PATH = Path(__file__).with_name("aggregation_prompt.md")


def _normalized_identity(value: str) -> str:
    """Normalize harmless casing and whitespace differences only."""
    return " ".join(value.casefold().split())


def _premerge_key(kind: str, item: Criterion | Attribute) -> tuple[str, ...]:
    payload = item.model_dump(
        mode="json",
        exclude={"id", "name", "description", "aliases"},
    )
    return (
        kind,
        _normalized_identity(item.id),
        _normalized_identity(item.name),
        _normalized_identity(item.description),
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )


def deterministic_premerge(
    collections: list[CriteriaAttributeSet],
) -> list[CriteriaAttributeSet]:
    """Remove only obvious duplicate items before semantic aggregation.

    Items merge only when their kind, normalized ID, normalized name,
    normalized description, concrete schema, and all non-text semantic fields
    agree. Anything less certain stays separate for the aggregation model.
    """
    seen: dict[tuple[str, ...], Criterion | Attribute] = {}
    output: list[CriteriaAttributeSet] = []

    for collection in collections:
        criteria: list[Criterion] = []
        attributes: list[Attribute] = []
        for kind, source, target in (
            ("criterion", collection.criteria, criteria),
            ("attribute", collection.attributes, attributes),
        ):
            for original in source:
                item = original.model_copy(deep=True)
                key = _premerge_key(kind, item)
                existing = seen.get(key)
                if existing is None:
                    seen[key] = item
                    target.append(item)  # type: ignore[arg-type]
                    continue

                aliases = list(existing.aliases)
                for alias in item.aliases:
                    if alias != existing.name and alias not in aliases:
                        aliases.append(alias)
                existing.aliases = aliases

        output.append(
            CriteriaAttributeSet(
                source_item_ids=list(collection.source_item_ids),
                criteria=criteria,
                attributes=attributes,
            )
        )
    return output


def build_market_aggregation_graph(
    model: BaseChatModel,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    return create_agent(
        model=model,
        tools=[],
        system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
        response_format=ToolStrategy(CriteriaAttributeSet),
        name="market_aggregation_agent",
    )


class MarketAggregationAgent:
    """Canonicalize all researched collections in one model call."""

    def __init__(self, model: BaseChatModel) -> None:
        self.graph = build_market_aggregation_graph(model)

    @staticmethod
    def _input(collections: list[CriteriaAttributeSet]) -> dict[str, Any]:
        if not collections:
            raise ValueError("aggregation requires at least one collection")
        context = [collection.model_dump(mode="json") for collection in collections]
        return {
            "messages": [
                HumanMessage(
                    content=json.dumps(
                        {"researched_collections": context},
                        ensure_ascii=False,
                    )
                )
            ]
        }

    @staticmethod
    def _result(
        state: dict[str, Any],
        collections: list[CriteriaAttributeSet],
    ) -> CriteriaAttributeSet:
        raw = state["structured_response"]
        result = (
            raw
            if isinstance(raw, CriteriaAttributeSet)
            else CriteriaAttributeSet.model_validate(raw)
        )
        source_item_ids = list(
            dict.fromkeys(
                item_id
                for collection in collections
                for item_id in collection.source_item_ids
            )
        )
        return result.model_copy(update={"source_item_ids": source_item_ids})

    def invoke(
        self,
        collections: list[CriteriaAttributeSet],
    ) -> CriteriaAttributeSet:
        state = self.graph.invoke(self._input(collections))
        return self._result(state, collections)

    async def ainvoke(
        self,
        collections: list[CriteriaAttributeSet],
    ) -> CriteriaAttributeSet:
        state = await self.graph.ainvoke(self._input(collections))
        return self._result(state, collections)
