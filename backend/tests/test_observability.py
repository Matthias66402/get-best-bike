from opentelemetry.sdk.trace import TracerProvider

from app import observability
from app.observability import chat_turn, observation_name, observe, tracing_enabled


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
    assert not observability._enabled
