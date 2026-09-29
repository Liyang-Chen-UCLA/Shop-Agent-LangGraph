from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from ...core.llm import build_deepseek_model
from ...domain.market_cache import MarketCache
from .draft import RelationDraft
from .schemas import RelationGraph, profile_hash, validate_graph
from .store import RelationStore, RevisionConflict
from .tools import build_relation_tools

PROMPT_PATH = Path(__file__).with_name("prompt.md")


class RelationAgent:
    def __init__(self, model: BaseChatModel | None = None, *, store=None, market_cache=None, search=None):
        self.model = model
        self.store = store if store is not None else RelationStore()
        self.market_cache = market_cache if market_cache is not None else MarketCache()
        self.search = search

    def invoke(self, node_id: str, *, refresh: bool = False) -> RelationGraph:
        node = self.market_cache.node(node_id)
        profile = self.market_cache.load(node)
        if profile is None:
            raise ValueError("a completed, persisted Market Profile is required before relation research")
        fingerprint = profile_hash(profile)
        previous = self.store.load(node_id)
        if previous and previous.profile_hash == fingerprint and not refresh:
            validate_graph(previous, profile)
            return previous
        now = datetime.now(timezone.utc)
        base_revision = previous.revision if previous else 0
        if previous and previous.profile_hash == fingerprint:
            graph = previous.model_copy(deep=True, update={"revision": base_revision + 1, "updated_at": now})
        else:
            # A changed profile invalidates even same-ID claims. Supply the old graph
            # for review, but require explicit re-addition against the new profile.
            graph = RelationGraph(node_id=node_id, profile_hash=fingerprint,
                revision=base_revision + 1, created_at=previous.created_at if previous else now, updated_at=now)
        draft = RelationDraft(graph, profile)
        agent = create_agent(
            model=self.model or build_deepseek_model(),
            tools=build_relation_tools(draft, self.search),
            system_prompt=PROMPT_PATH.read_text(encoding="utf-8"), name="relation_agent",
        )
        context = {"node": node.model_dump(), "market_profile": profile.model_dump(mode="json"),
                   "draft": graph.model_dump(mode="json"),
                   "previous_graph_for_review": previous.model_dump(mode="json") if previous else None,
                   "profile_changed": bool(previous and previous.profile_hash != fingerprint)}
        agent.invoke({"messages": [{"role": "user", "content": json.dumps(context, ensure_ascii=False)}]},
                     config={"recursion_limit": 100})
        if not draft.finalized:
            raise RuntimeError("Relation Agent stopped without finalizing; no graph was saved")
        current_profile = self.market_cache.load(node)
        if current_profile is None or profile_hash(current_profile) != fingerprint:
            raise RevisionConflict("Market Profile changed during research; rerun on the new profile")
        validate_graph(draft.graph, current_profile)
        self.store.save(draft.graph, base_revision)
        return draft.graph

    async def ainvoke(self, node_id: str, *, refresh: bool = False) -> RelationGraph:
        return await asyncio.to_thread(self.invoke, node_id, refresh=refresh)


def build_relation_agent(model: BaseChatModel | None = None) -> RelationAgent:
    return RelationAgent(model)


class LazyRelationAgent:
    # Construct per invocation: fresh env paths and isolated mutable drafts.
    def invoke(self, node_id: str, *, refresh: bool = False) -> RelationGraph:
        return build_relation_agent().invoke(node_id, refresh=refresh)

    async def ainvoke(self, node_id: str, *, refresh: bool = False) -> RelationGraph:
        return await build_relation_agent().ainvoke(node_id, refresh=refresh)


relation_agent = LazyRelationAgent()
