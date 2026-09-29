from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from ..relation.store import RevisionConflict
from .schemas import PersonalizationSession

DEFAULT_DIRECTORY = Path(__file__).resolve().parents[4] / ".cache" / "personalization"


class PersonalizationStore:
    def __init__(self, directory: Path | None = None):
        self.directory = directory if directory is not None else Path(
            os.getenv("PERSONALIZATION_CACHE_DIR", str(DEFAULT_DIRECTORY)))

    def path(self, thread_id: str, task_id: str) -> Path:
        if not thread_id.strip() or not task_id.strip():
            raise ValueError("thread_id and task_id must not be empty")
        # User-controlled identifiers never become filesystem path components.
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        return self.directory / digest(thread_id) / f"{digest(task_id)}.json"

    def load(self, thread_id: str, task_id: str) -> PersonalizationSession | None:
        try:
            session = PersonalizationSession.model_validate_json(
                self.path(thread_id, task_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if (session.thread_id, session.task_id) != (thread_id, task_id):
            raise ValueError("personalization session identity mismatch")
        return session

    def save(self, session: PersonalizationSession, base_revision: int) -> None:
        path = self.path(session.thread_id, session.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RevisionConflict("personalization is being updated; retry later") from error
        os.close(descriptor)
        temporary = None
        try:
            current = self.load(session.thread_id, session.task_id)
            if (current.revision if current else 0) != base_revision:
                raise RevisionConflict("personalization changed; reload before updating")
            if session.revision != base_revision + 1:
                raise ValueError("session revision must increment by one")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             delete=False) as output:
                temporary = Path(output.name)
                output.write(session.model_dump_json(indent=2))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)
