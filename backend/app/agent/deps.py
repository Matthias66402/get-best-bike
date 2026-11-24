from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings


@dataclass
class AgentDeps:
    """Abhängigkeiten, die jedes Tool über ``ctx.deps`` erhält."""

    settings: Settings
    http: httpx.AsyncClient
    # Factory statt Session: Tools können parallel laufen, eine AsyncSession ist nicht nebenläufig nutzbar.
    session_factory: async_sessionmaker[AsyncSession]
    user_name: str
    today: date = field(default_factory=date.today)
    # Vorgaben aus dem Profil (Körpermaße für die Rahmengröße)
    height_cm: int | None = None
    inseam_cm: int | None = None
    # Hinweis aus dem Intake für die Anweisungen des Beraters in diesem Lauf (erkannte Kriterien); nicht in der
    # Nutzernachricht, weil er nur für die aktuelle Nachricht gilt und spätere Läufe ihn sonst als Nutzertext läsen
    intake_note: str | None = None
    # Vom Intake erkannte Kriterien dieser Nachricht (z. B. use_case, budget_eur), für den Output-Validator
    criteria: dict[str, Any] = field(default_factory=dict)
    # Intake hat Websuchen geplant: Diese Nachricht verlangt eine (neue) Rangliste
    ranking_due: bool = False
    # Zusätzliche Werte für Tests/Erweiterungen
    extra: dict[str, Any] = field(default_factory=dict)
