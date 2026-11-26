"""Tracing mit Langfuse oder LangSmith (TRACING_BACKEND).

Pydantic AI erzeugt OpenTelemetry-Spans für Agent-Läufe, Modellaufrufe (Modell, Tokens), Tool-Aufrufe und
Embeddings; das gewählte Backend exportiert sie. Je Chat-Nachricht entsteht ein Trace mit der Wurzel `chat-turn`
(Eingabe: Nutzernachricht, Ausgabe: Antwort), je Beratung eine Session (Langfuse) bzw. ein Thread (LangSmith).
Modell- und Embedding-Spans bekommen feste Namen je Agent statt des Modellnamens (`observation_name`).

- langfuse: aktiv mit LANGFUSE_PUBLIC_KEY und LANGFUSE_SECRET_KEY (Basis-URL: LANGFUSE_BASE_URL, Umgebung:
  LANGFUSE_TRACING_ENVIRONMENT); LANGFUSE_TRACING_ENABLED=false schaltet es ab. Export über das Langfuse-SDK.
- langsmith: aktiv mit LANGSMITH_API_KEY (Projekt: LANGSMITH_PROJECT, Region: LANGSMITH_ENDPOINT);
  LANGSMITH_TRACING=false schaltet es ab. Export per OTLP an `<LANGSMITH_ENDPOINT>/otel/v1/traces`.

Die Tests schalten beide ab (tests/conftest.py).
"""

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.trace import Status, StatusCode
from pydantic_ai import Agent, Embedder

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# "langfuse", "langsmith" oder None (aus)
_backend: str | None = None
# Nur bei LangSmith: eigener Provider (Export) und Tracer für eigene Observations
_provider: TracerProvider | None = None
_tracer: trace.Tracer | None = None

# Observation-Typen (wie bei Langfuse) als Run-Typen von LangSmith
LANGSMITH_RUN_TYPES = {
    "span": "chain", "chain": "chain", "agent": "chain", "tool": "tool", "retriever": "retriever",
    "generation": "llm", "embedding": "embedding",
}  # fmt: skip


def tracing_enabled() -> bool:
    return _backend is not None


def setup_tracing() -> None:
    """Einmal beim Start: Export zum gewählten Backend einrichten und Pydantic AI instrumentieren."""
    global _backend
    settings = get_settings()
    if settings.tracing_backend == "langfuse":
        if not (settings.langfuse_tracing_enabled and settings.langfuse_public_key and settings.langfuse_secret_key):
            return
        from langfuse import Langfuse

        # Zugangsdaten, Basis-URL und Umgebung liest das SDK aus den LANGFUSE_*-Umgebungsvariablen
        Langfuse()
    elif settings.tracing_backend == "langsmith":
        if not (settings.langsmith_tracing and settings.langsmith_api_key):
            return
        _setup_langsmith(settings)
    else:
        return
    _add_span_renamer()
    Agent.instrument_all()
    Embedder.instrument_all()
    _backend = settings.tracing_backend
    logger.info("Tracing aktiv: %s", _backend)


def _setup_langsmith(settings: Settings) -> None:
    global _provider, _tracer
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = OTLPSpanExporter(
        endpoint=f"{settings.langsmith_endpoint.rstrip('/')}/otel/v1/traces",
        headers={"x-api-key": settings.langsmith_api_key, "Langsmith-Project": settings.langsmith_project},
    )
    _provider = TracerProvider(resource=Resource.create({"service.name": "get-best-bike"}))
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    _tracer = _provider.get_tracer("get-best-bike")


def observation_name(name: str, attributes: Mapping[str, Any], parent_attributes: Mapping[str, Any]) -> str | None:
    """Neuer Name für Modell- und Embedding-Spans von Pydantic AI, sonst None.

    Pydantic AI benennt sie nach dem Modell („chat gpt-5-mini“, „embeddings text-embedding-3-small“); ein
    Modellwechsel änderte dann alle Namen, Filter und Evaluatoren im Tracing. Das Modell steht ohnehin an der
    Generation bzw. am LLM-Run. Neu: je Agent ein fester Name, z. B. „bike-recommender llm-call“.
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
    # Provider des Langfuse-Clients bzw. der eigene für LangSmith; exportiert wird erst am Span-Ende
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(_RenameSpans())


def _attribute(value: Any) -> str | bool | int | float:
    """OpenTelemetry-Attribute sind nur einfache Werte; alles andere als JSON."""
    if isinstance(value, str | bool | int | float):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _io_value(key: str, value: Any) -> str:
    """Ein- oder Ausgabe für LangSmith als JSON-Objekt: Dicts direkt, sonst unter `input` bzw. `output`."""
    return json.dumps(value if isinstance(value, dict) else {key: value}, ensure_ascii=False, default=str)


def _metadata_key(key: str) -> str:
    prefix = "langsmith.metadata" if _backend == "langsmith" else "langfuse.observation.metadata"
    return f"{prefix}.{key}"


def langsmith_attributes(
    as_type: str = "span", *, input: Any = None, metadata: Mapping[str, Any] | None = None, **_: Any
) -> dict[str, str | bool | int | float]:
    """Start-Attribute einer eigenen Observation für LangSmith (Run-Typ, Eingabe, Metadaten)."""
    attributes: dict[str, str | bool | int | float] = {"langsmith.span.kind": LANGSMITH_RUN_TYPES.get(as_type, "chain")}
    if input is not None:
        attributes["input.value"] = _io_value("input", input)
    for key, value in (metadata or {}).items():
        attributes[f"langsmith.metadata.{key}"] = _attribute(value)
    return attributes


def add_metadata(**values: str | int | float) -> None:
    """Metadaten an der aktuellen Observation, z. B. am Modell-Span von Pydantic AI."""
    if not _backend:
        return
    span = trace.get_current_span()
    if span.is_recording():
        for key, value in values.items():
            span.set_attribute(_metadata_key(key), value)


def _set_warning(span: trace.Span, message: str | None) -> None:
    if _backend == "langsmith":
        # LangSmith kennt kein Level WARNING; ein Fehlerstatus wäre falsch, der Lauf geht ja weiter
        span.set_attribute("langsmith.metadata.level", "WARNING")
        if message:
            span.set_attribute("langsmith.metadata.status_message", message)
    else:
        span.set_attribute("langfuse.observation.level", "WARNING")
        if message:
            span.set_attribute("langfuse.observation.status_message", message)


def mark_warning(message: str) -> None:
    """Markiert die aktuelle Observation (z. B. den Tool-Span von Pydantic AI) als Warnung, damit gescheiterte
    Tool-Aufrufe im Tracing filterbar sind; der Lauf selbst geht weiter."""
    if not _backend:
        return
    span = trace.get_current_span()
    if span.is_recording():
        _set_warning(span, message)


def shutdown_tracing() -> None:
    """Gepufferte Spans beim Beenden senden."""
    if _backend == "langfuse":
        from langfuse import get_client

        get_client().shutdown()
    elif _provider is not None:
        _provider.shutdown()


class _NoopObservation:
    def update(self, **_: Any) -> None:
        pass


class _SpanObservation:
    """Eigene Observation bei LangSmith mit derselben `update()`-Schnittstelle wie bei Langfuse."""

    def __init__(self, span: trace.Span) -> None:
        self.span = span

    def update(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
        **_: Any,
    ) -> None:
        if output is not None:
            self.span.set_attribute("output.value", _io_value("output", output))
        for key, value in (metadata or {}).items():
            self.span.set_attribute(f"langsmith.metadata.{key}", _attribute(value))
        if level == "ERROR":
            self.span.set_status(Status(StatusCode.ERROR, status_message))
        elif level == "WARNING":
            _set_warning(self.span, status_message)


@contextmanager
def observe(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Eigene Observation (z. B. `retriever`) im aktuellen Trace; ohne Tracing ein Platzhalter mit `update()`."""
    if _backend == "langsmith" and _tracer is not None:
        with _tracer.start_as_current_span(name, attributes=langsmith_attributes(as_type, **kwargs)) as span:
            yield _SpanObservation(span)
        return
    if _backend != "langfuse":
        yield _NoopObservation()
        return
    from langfuse import get_client

    with get_client().start_as_current_observation(name=name, as_type=as_type, **kwargs) as observation:
        yield observation


@contextmanager
def chat_turn(*, message: str, user_id: int, consultation_id: int) -> Iterator[Any]:
    """Wurzel eines Traces je Chat-Nachricht; User und Session (bzw. Thread) gelten für die ganze Beratung."""
    if _backend == "langsmith":
        # LangSmith gruppiert Threads über die Metadaten thread_id/session_id. langsmith.trace.session_id wäre
        # falsch: „Session“ heißt in LangSmith das Projekt.
        metadata = {
            "thread_id": f"consultation-{consultation_id}",
            "user_id": str(user_id),
            "consultation_id": str(consultation_id),
        }
        with observe("chat-turn", as_type="span", input=message, metadata=metadata) as root:
            root.span.set_attribute("langsmith.trace.name", "chat-turn")
            root.span.set_attribute("langsmith.span.tags", "chat")
            yield root
        return
    if _backend != "langfuse":
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
