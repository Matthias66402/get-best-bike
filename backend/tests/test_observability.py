import json
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from app import observability
from app.observability import add_metadata, chat_turn, mark_warning, observation_name, observe, tracing_enabled


def test_model_spans_named_by_agent_not_model() -> None:
    chat = {"gen_ai.operation.name": "chat"}
    assert observation_name("chat gpt-5-mini", chat, {"gen_ai.agent.name": "bike-recommender"}) == (
        "bike-recommender llm-call"
    )
    assert observation_name("chat gpt-4o-mini", chat, {"agent_name": "intake"}) == "intake llm-call"
    assert observation_name("chat gpt-4o-mini", chat, {}) == "llm-call"
    embeddings = {"gen_ai.operation.name": "embeddings"}
    assert observation_name("embeddings text-embedding-3-small", embeddings, {}) == "embed-text"
    assert observation_name("intake run", {}, {}) is None
    assert observation_name("running tool", {"gen_ai.operation.name": "execute_tool"}, {}) is None


def test_rename_survives_later_update_name() -> None:
    """Pydantic AI benennt den Modell-Span nach dem Start erneut; der feste Name muss bleiben."""
    provider = TracerProvider()
    provider.add_span_processor(observability._RenameSpans())
    tracer = provider.get_tracer("test")
    span = tracer.start_span(
        "chat gpt-5-mini", attributes={"gen_ai.operation.name": "chat", "gen_ai.agent.name": "intake"}
    )
    assert span.name == "intake llm-call"
    span.update_name("chat gpt-5-mini-2025-08-07")
    span.end()
    assert span.name == "intake llm-call"
    other = tracer.start_span("intake run")
    other.update_name("intake run 2")
    other.end()
    assert other.name == "intake run 2"


def test_tracing_disabled_in_tests() -> None:
    assert not tracing_enabled()


def test_observations_are_noops_without_tracing() -> None:
    with chat_turn(message="Gravelbike bis 2000 €", user_id=1, consultation_id=2) as root:
        root.update(output="…", metadata={"outcome": "recommendation"})
        with observe("retrieve-knowledge", as_type="retriever") as span:
            span.update(output=[])
    assert observability._backend is None


@pytest.fixture
def langsmith_spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """LangSmith-Backend mit lokalem Provider: Spans landen im Speicher statt bei LangSmith."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_backend", "langsmith")
    monkeypatch.setattr(observability, "_tracer", provider.get_tracer("test"))
    yield exporter


def test_langsmith_chat_turn_is_thread_root(langsmith_spans: InMemorySpanExporter) -> None:
    with chat_turn(message="Gravelbike bis 2000 €", user_id=1, consultation_id=2) as root:
        root.update(output="Hier sind fünf Räder", metadata={"outcome": "recommendation", "tool_calls": 3})
    (span,) = langsmith_spans.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert span.name == "chat-turn"
    assert attributes["langsmith.trace.name"] == "chat-turn"
    assert attributes["langsmith.span.kind"] == "chain"
    assert attributes["langsmith.span.tags"] == "chat"
    assert attributes["langsmith.metadata.thread_id"] == "consultation-2"
    assert attributes["langsmith.metadata.user_id"] == "1"
    assert attributes["langsmith.metadata.outcome"] == "recommendation"
    assert attributes["langsmith.metadata.tool_calls"] == 3
    assert json.loads(str(attributes["input.value"])) == {"input": "Gravelbike bis 2000 €"}
    assert json.loads(str(attributes["output.value"])) == {"output": "Hier sind fünf Räder"}


def test_langsmith_observation_kinds_and_levels(langsmith_spans: InMemorySpanExporter) -> None:
    with observe("retrieve-knowledge", as_type="retriever", input="Gravel", metadata={"top_k": 5}) as retrieval:
        retrieval.update(output=[{"document": "Ratgeber", "score": 0.8}])
        add_metadata(rate_limit_wait_s=1.5)
    with observe("search_bikes", as_type="tool", input={"query": "Gravel"}):
        mark_warning("Tavily hat nicht geantwortet")
    with observe("retrieve-knowledge", as_type="retriever") as failed:
        failed.update(level="ERROR", status_message="DB nicht erreichbar")
    retrieval_span, tool_span, failed_span = langsmith_spans.get_finished_spans()

    retrieval_attributes = dict(retrieval_span.attributes or {})
    assert retrieval_attributes["langsmith.span.kind"] == "retriever"
    assert retrieval_attributes["langsmith.metadata.top_k"] == 5
    assert retrieval_attributes["langsmith.metadata.rate_limit_wait_s"] == 1.5
    assert json.loads(str(retrieval_attributes["output.value"])) == {"output": [{"document": "Ratgeber", "score": 0.8}]}

    tool_attributes = dict(tool_span.attributes or {})
    assert tool_attributes["langsmith.span.kind"] == "tool"
    assert json.loads(str(tool_attributes["input.value"])) == {"query": "Gravel"}
    assert tool_attributes["langsmith.metadata.level"] == "WARNING"
    assert tool_span.status.status_code == StatusCode.UNSET  # Warnung, kein Fehler

    assert failed_span.status.status_code == StatusCode.ERROR
    assert failed_span.status.description == "DB nicht erreichbar"
