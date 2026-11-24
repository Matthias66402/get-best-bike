"""Tracing mit Langfuse.

Pydantic AI erzeugt OpenTelemetry-Spans für Agent-Läufe, Modellaufrufe (als Generation mit Modell, Tokens, Kosten),
Tool-Aufrufe und Embeddings; das Langfuse-SDK exportiert sie. Je Chat-Nachricht entsteht ein Trace mit der Wurzel
`chat-turn` (Eingabe: Nutzernachricht, Ausgabe: Antwort), je Beratung eine Session.

Aktiv nur mit LANGFUSE_PUBLIC_KEY und LANGFUSE_SECRET_KEY (Basis-URL: LANGFUSE_BASE_URL, Umgebung:
LANGFUSE_TRACING_ENVIRONMENT); LANGFUSE_TRACING_ENABLED=false schaltet es ab, z. B. in den Tests.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic_ai import Agent, Embedder

from app.config import get_settings

logger = logging.getLogger(__name__)

_enabled = False


def tracing_enabled() -> bool:
    return _enabled


def setup_tracing() -> None:
    """Einmal beim Start: Langfuse-Client anlegen und Pydantic AI instrumentieren."""
    global _enabled
    settings = get_settings()
    if not (settings.langfuse_tracing_enabled and settings.langfuse_public_key and settings.langfuse_secret_key):
        return
    from langfuse import Langfuse

    # Zugangsdaten, Basis-URL und Umgebung liest das SDK aus den LANGFUSE_*-Umgebungsvariablen
    Langfuse()
    Agent.instrument_all()
    Embedder.instrument_all()
    _enabled = True
    logger.info("Langfuse-Tracing aktiv")


def shutdown_tracing() -> None:
    """Gepufferte Spans beim Beenden senden."""
    if _enabled:
        from langfuse import get_client

        get_client().shutdown()


class _NoopObservation:
    def update(self, **_: Any) -> None:
        pass


@contextmanager
def observe(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Eigene Observation (z. B. `retriever`) im aktuellen Trace; ohne Tracing ein Platzhalter mit `update()`."""
    if not _enabled:
        yield _NoopObservation()
        return
    from langfuse import get_client

    with get_client().start_as_current_observation(name=name, as_type=as_type, **kwargs) as observation:
        yield observation


@contextmanager
def chat_turn(*, message: str, user_id: int, consultation_id: int) -> Iterator[Any]:
    """Wurzel eines Traces je Chat-Nachricht; User und Session gelten für alle Observations darin."""
    if not _enabled:
        yield _NoopObservation()
        return
    from langfuse import propagate_attributes

    with (
        observe("chat-turn", as_type="span", input=message) as root,
        propagate_attributes(
            user_id=str(user_id),
            session_id=f"consultation-{consultation_id}",
            trace_name="chat-turn",
            tags=["chat"],
            metadata={"consultation_id": str(consultation_id)},
        ),
    ):
        yield root
