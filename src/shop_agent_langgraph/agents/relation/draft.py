from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock

from ..market.schemas import MarketResult
from .schemas import RelationGraph, validate_graph


class RelationDraft:
    """One invocation's isolated draft; every edit is validated before it commits."""

    def __init__(self, graph: RelationGraph, profile: MarketResult):
        self.graph = graph.model_copy(deep=True)
        self.profile = profile
        self.finalized = False
        self.search_count = 0
        self.completed_searches = 0
        self.pending_searches = 0
        self.sources: dict[str, list[dict]] = {}
        self.lock = RLock()

    def upsert(self, field: str, item) -> dict:
        with self.lock:
            if self.finalized:
                raise ValueError("draft already finalized")
            candidate = self.graph.model_copy(deep=True)
            items = getattr(candidate, field)
            existing = next((i for i, old in enumerate(items) if old.id == item.id), None)
            if existing is None:
                items.append(item)
            else:
                items[existing] = item
            validate_graph(candidate, self.profile)
            self.graph = candidate
            return {"saved": item.id, "field": field}

    def retire(self, claim_id: str) -> dict:
        with self.lock:
            for field in ("relations", "utility_rules"):
                for claim in getattr(self.graph, field):
                    if claim.id == claim_id:
                        updated = type(claim).model_validate({**claim.model_dump(), "status": "retired"})
                        return self.upsert(field, updated)
            raise ValueError("unknown claim ID")

    def finalize(self) -> dict:
        with self.lock:
            if not self.completed_searches or self.pending_searches:
                raise ValueError("complete evidence searches before finalizing")
            validate_graph(self.graph, self.profile)
            if not (self.graph.relations or self.graph.utility_rules or self.graph.unresolved):
                raise ValueError("record findings or an unresolved question before finalizing")
            self.graph.updated_at = datetime.now(timezone.utc)
            self.finalized = True
            return {"finalized": True, "revision": self.graph.revision,
                    "relations": len(self.graph.relations), "utility_rules": len(self.graph.utility_rules)}
