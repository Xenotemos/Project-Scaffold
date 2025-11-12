"""Persistence helpers for writing memories to the database."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from sqlmodel import Session, SQLModel, create_engine, select

from .models import LongTermMemoryRecord


class MemoryRepository:
    """Encapsulates long-term memory storage using SQLite."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path or Path(__file__).resolve().parent / "memory.db"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        database_uri = f"sqlite:///{self.database_path}"
        self._engine = create_engine(database_uri, echo=False)
        SQLModel.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Context manager yielding a SQLModel session."""
        with Session(self._engine) as session:
            yield session

    def save_record(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        """Persist a record and return the refreshed instance."""
        with self.session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def recent(self, limit: int = 5) -> Sequence[LongTermMemoryRecord]:
        """Fetch the most recent long-term memories."""
        statement = (
            select(LongTermMemoryRecord)
            .order_by(LongTermMemoryRecord.created_at.desc())
            .limit(limit)
        )
        with self.session() as session:
            return list(session.exec(statement))

    def dispose(self) -> None:
        """Release database connections to allow filesystem cleanup."""
        self._engine.dispose()
