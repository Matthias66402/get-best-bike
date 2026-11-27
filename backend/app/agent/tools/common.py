import re
from typing import Any

import httpx

from app.observability import mark_warning


def tool_error(message: str) -> dict[str, Any]:
    """Fehler als Tool-Ergebnis: Das LLM kann darauf reagieren (andere Suche, Rückfrage, Hinweis).

    Im Tracing wird der Tool-Aufruf als WARNING markiert (z. B. Filter „gescheiterte Suchen“ in Langfuse)."""
    mark_warning(message)
    return {"error": message}


def http_error_message(service: str, exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return f"{service} hat nicht rechtzeitig geantwortet. Später erneut versuchen oder ohne diese Daten beraten."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{service} meldet HTTP {exc.response.status_code}: {exc.response.text[:200]}"
    return f"{service} ist nicht erreichbar: {exc}"


def words(text: str) -> list[str]:
    """Kleingeschriebene Wörter ohne Satz- und Sonderzeichen ("E‑Power Easy" → ["e", "power", "easy"])."""
    return re.findall(r"[0-9a-zäöüß]+", text.casefold())


def model_family(model: str) -> str | None:
    """Erstes aussagekräftiges Wort des Modellnamens, z. B. "talon" in "Talon E+ EX Gen2"."""
    return next((w for w in words(model) if len(w) >= 3 and not w.isdigit()), None)
