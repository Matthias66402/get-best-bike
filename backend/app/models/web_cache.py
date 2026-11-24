from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class WebCacheEntry(Base):
    """Zwischengespeicherte Tavily-Antwort (Suche oder Seiteninhalt), damit gleiche Anfragen kein Kontingent kosten."""

    __tablename__ = "web_cache"

    # z. B. "search:gravel bike 2000 euro|6" oder "extract:https://..."
    key: Mapped[str] = mapped_column(String(2048), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
