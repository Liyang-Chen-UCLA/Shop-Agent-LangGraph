from __future__ import annotations

import os
from typing import Any

from langchain.tools import tool
from langchain_tavily import TavilySearch

from .schemas import ResearchResult


@tool(args_schema=ResearchResult)
def submit_research_result(
    item_id: str,
    evidence: list[Any],
) -> dict[str, Any]:
    """Submit one product's metric and attribute evidence."""
    return ResearchResult(
        item_id=item_id,
        evidence=evidence,
    ).model_dump()


def build_research_tools() -> list[Any]:
    return [
        TavilySearch(
            max_results=3,
            topic="general",
            api_key=os.getenv("TAVILY_API_KEY"),
        ),
    ]
