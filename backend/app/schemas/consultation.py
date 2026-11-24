from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models import ConsultationStatus


class ConsultationCreate(BaseModel):
    title: str = Field(default="Neue Beratung", min_length=1, max_length=200)


class ConsultationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)


class ConsultationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    status: ConsultationStatus
    created_at: datetime
    updated_at: datetime


class ConsultationListItem(ConsultationSummary):
    """Eintrag der Übersichtsliste; `bikes` (Marke und Modell der Rangliste) für Suche und Vorschau."""

    bikes: list[str] = Field(default_factory=list)


def ranked_bikes(recommendation: dict[str, Any] | None) -> list[str]:
    if not recommendation:
        return []
    return [
        f"{bike.get('brand', '')} {bike.get('model', '')}".strip()
        for bike in recommendation.get("ranking") or []
        if isinstance(bike, dict)
    ]


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    tool_calls: list[Any]
    created_at: datetime


class ConsultationRead(ConsultationSummary):
    recommendation: dict[str, Any] | None
    messages: list[ChatMessageRead]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
