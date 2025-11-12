"""SQLModel models representing long-term memories."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel


class LongTermMemoryRecord(SQLModel, table=True):
    """Persistent record describing a consolidated memory."""

    id: Optional[int] = Field(default=None, primary_key=True)
    content: str = Field(index=True, min_length=1)
    mood: Optional[str] = Field(default=None, index=True)
    hormone_snapshot: dict[str, float] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
# NOTE: `metadata` is reserved in SQLAlchemy declarative models, so use
# `attributes` to store arbitrary structured data.
    attributes: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)
