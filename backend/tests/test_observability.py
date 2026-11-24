from app import observability
from app.observability import chat_turn, observe, tracing_enabled


def test_tracing_disabled_in_tests() -> None:
    assert not tracing_enabled()


def test_observations_are_noops_without_tracing() -> None:
    with chat_turn(message="Gravelbike bis 2000 €", user_id=1, consultation_id=2) as root:
        root.update(output="…", metadata={"outcome": "recommendation"})
        with observe("retrieve-knowledge", as_type="retriever") as span:
            span.update(output=[])
    assert not observability._enabled
