from typing import Any

import httpx


def tool_error(message: str) -> dict[str, Any]:
    """Fehler als Tool-Ergebnis: Das LLM kann darauf reagieren (andere Suche, Rückfrage, Hinweis)."""
    return {"error": message}


def http_error_message(service: str, exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return f"{service} hat nicht rechtzeitig geantwortet. Später erneut versuchen oder ohne diese Daten beraten."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{service} meldet HTTP {exc.response.status_code}: {exc.response.text[:200]}"
    return f"{service} ist nicht erreichbar: {exc}"
