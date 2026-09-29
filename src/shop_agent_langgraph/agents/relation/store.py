from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .schemas import RelationGraph

DEFAULT_DIRECTORY = Path(__file__).resolve().parents[4] / ".cache" / "relation_nodes"


class RevisionConflict(RuntimeError):
    pass


class RelationStore:
    def __init__(self, directory: Path | None = None):
        configured = os.getenv("RELATION_CACHE_DIR")
        self.directory = directory if directory is not None else (
            Path(configured).expanduser().resolve() if configured else DEFAULT_DIRECTORY
        )

    def _directory(self, node_id: str) -> Path:
        if not node_id.isascii() or not node_id.isdecimal():
            raise ValueError("node ID must be numeric")
        return self.directory / node_id

    def load(self, node_id: str) -> RelationGraph | None:
        try:
            graph = RelationGraph.model_validate_json(
                (self._directory(node_id) / "current.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return None
        if graph.node_id != node_id:
            raise ValueError("stored graph node ID mismatch")
        return graph

    def save(self, graph: RelationGraph, base_revision: int) -> None:
        directory = self._directory(graph.node_id)
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / ".write.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RevisionConflict("another writer holds the graph lock; retry later") from error
        os.close(descriptor)
        temporary = None
        try:
            current = self.load(graph.node_id)
            if (current.revision if current else 0) != base_revision:
                raise RevisionConflict("graph changed during research; reload before updating")
            if graph.revision != base_revision + 1:
                raise ValueError("revision must increment by one")
            if current and graph.created_at != current.created_at:
                raise ValueError("updates must preserve created_at")
            revisions = directory / "revisions"
            revisions.mkdir(exist_ok=True)
            payload = graph.model_dump_json(indent=2)
            # Exclusive history creation also prevents silent reuse of an orphan
            # revision left by a process interruption before current.json replacement.
            with (revisions / f"{graph.revision}.json").open("x", encoding="utf-8") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as output:
                temporary = Path(output.name)
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, directory / "current.json")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)
