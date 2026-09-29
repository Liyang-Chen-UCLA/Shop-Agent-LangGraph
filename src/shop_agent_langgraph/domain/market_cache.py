from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, model_validator

from ..agents.market.schemas import MarketResult
from ..agents.route.schemas import TaxonomyNode
from .criteria import SchemaModel
from .taxonomy import get_nodes


logger = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "market_nodes"


class CachedMarketResult(SchemaModel):
    schema_version: Literal[2] = 2
    node: TaxonomyNode
    saved_at: datetime
    result: MarketResult

    @model_validator(mode="after")
    def require_completed(self) -> CachedMarketResult:
        if self.result.status != "completed":
            raise ValueError("only completed market results can be cached")
        return self


class MarketCache:
    """Persist completed market definitions by canonical taxonomy node ID."""

    def __init__(self, directory: Path | None = None) -> None:
        configured = os.getenv("MARKET_CACHE_DIR")
        self.directory = directory if directory is not None else (
            Path(configured).expanduser().resolve() if configured else DEFAULT_CACHE_DIR
        )

    @staticmethod
    def node(node_id: str) -> TaxonomyNode:
        records, missing = get_nodes([node_id])
        if missing:
            raise ValueError(f"unknown taxonomy node ID: {node_id}")
        record = records[0]
        return TaxonomyNode(**{key: record[key] for key in (
            "node_id", "node_name", "node_path",
        )})

    def _path(self, node: TaxonomyNode) -> Path:
        # Only canonical numeric IDs are allowed in filesystem paths.
        if not node.node_id.isascii() or not node.node_id.isdecimal():
            raise ValueError("taxonomy node ID must be numeric")
        return self.directory / f"{node.node_id}.json"

    def load(self, node: TaxonomyNode) -> MarketResult | None:
        path = self._path(node)
        try:
            cached = CachedMarketResult.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (ValidationError, UnicodeError):
            logger.warning("Ignoring invalid market cache: %s", path)
            return None
        if cached.node != node:
            logger.warning("Ignoring market cache with mismatched taxonomy: %s", path)
            return None
        return cached.result

    def save(self, node: TaxonomyNode, result: MarketResult) -> None:
        cached = CachedMarketResult(
            node=node, saved_at=datetime.now(timezone.utc), result=result,
        )
        path = self._path(node)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Write beside the destination and replace atomically, so readers never
        # observe a partly-written JSON file, including across server processes.
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory,
                prefix=f".{node.node_id}-", suffix=".tmp", delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(cached.model_dump_json(indent=2))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
