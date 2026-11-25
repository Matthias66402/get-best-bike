"""Tracing mit Langfuse.

Pydantic AI erzeugt OpenTelemetry-Spans für Agent-Läufe, Modellaufrufe (als Generation mit Modell, Tokens, Kosten),
Tool-Aufrufe und Embeddings; das Langfuse-SDK exportiert sie. Je Chat-Nachricht entsteht ein Trace mit der Wurzel
`chat-turn` (Eingabe: Nutzernachricht, Ausgabe: Antwort), je Beratung eine Session. Modell- und Embedding-Spans
bekommen feste Namen je Agent statt des Modellnamens (`observation_name`).

Aktiv nur mit LANGFUSE_PUBLIC_KEY und LANGFUSE_SECRET_KEY (Basis-URL: LANGFUSE_BASE_URL, Umgebung:
LANGFUSE_TRACING_ENVIRONMENT); LANGFUSE_TRACING_ENABLED=false schaltet es ab, z. B. in den Tests.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
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
    _add_span_renamer()
    Agent.instrument_all()
    Embedder.instrument_all()
    _enabled = True
    logger.info("Langfuse-Tracing aktiv")


def observation_name(name: str, attributes: Mapping[str, Any], parent_attributes: Mapping[str, Any]) -> str | None:
    """Neuer Name für Modell- und Embedding-Spans von Pydantic AI, sonst None.

    Pydantic AI benennt sie nach dem Modell („chat gpt-5-mini“, „embeddings text-embedding-3-small“); ein
    Modellwechsel änderte dann alle Namen, Filter und Evaluatoren in Langfuse. Das Modell steht ohnehin im Feld
    `model` der Generation. Neu: je Agent ein fester Name, z. B. „bike-recommender llm-call“.
    """
    operation = attributes.get("gen_ai.operation.name")
    if operation == "chat" and name.startswith("chat "):
        # Pydantic AI gibt den Agent-Namen per Baggage an den Span; sonst trägt ihn der Agent-Span darüber
        agent = (
            attributes.get("gen_ai.agent.name")
            or parent_attributes.get("gen_ai.agent.name")
            or parent_attributes.get("agent_name")
        )
        return f"{agent} llm-call" if agent else "llm-call"
    if operation == "embeddings" and name.startswith("embeddings "):
        return "embed-text"
    return None


class _RenameSpans(SpanProcessor):
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        parent = trace.get_current_span(parent_context)
        parent_attributes = dict(getattr(parent, "attributes", None) or {})
        original_update_name = span.update_name

        def rename(name: str) -> None:
            original_update_name(observation_name(name, span.attributes or {}, parent_attributes) or name)

        # Pydantic AI benennt den Modell-Span während des Aufrufs erneut („chat <Modell>“, sobald das Modell
        # feststeht); auch diese Umbenennung läuft so über observation_name
        span.update_name = rename  # type: ignore[method-assign]
        rename(span.name)


def _add_span_renamer() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):  # das SDK des Langfuse-Clients; exportiert wird erst am Span-Ende
        provider.add_span_processor(_RenameSpans())


def mark_warning(message: str) -> None:
    """Setzt die aktuelle Observation (z. B. den Tool-Span von Pydantic AI) auf Level WARNING, damit gescheiterte
    Tool-Aufrufe in Langfuse filterbar sind; der Lauf selbst geht weiter."""
    if not _enabled:
        return
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("langfuse.observation.level", "WARNING")
        span.set_attribute("langfuse.observation.status_message", message)


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
