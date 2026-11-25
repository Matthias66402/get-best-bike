"""Graph-Ablauf: Intake, parallele Vorab-Aufrufe (Websuche, Recherche), Absage bei fachfremden Nachrichten."""

from typing import Any

import httpx
import pytest
import respx
from pydantic_ai import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent.deps import AgentDeps
from app.agent.graph import OFF_TOPIC_MESSAGE, recommender_history, route_intake
from app.agent.intake import Prefetch, ResearchRequest, SearchRequest, intake_agent, plan_prefetch
from app.agent.recommendation_schema import BikeCategory
from app.agent.recommender import recommender_agent
from app.agent.runner import RunOutcome, run_recommender
from app.config import get_settings
from app.db import SessionLocal
from tests.helpers import SEARCH_OK, TAVILY_URL, recommendation, streaming_model

INTAKE = {
    "category": "trekking", "electric": True, "budget_eur": 3500, "use_case": "Pendeln", "height_cm": 175,
    "inseam_cm": 82, "search_queries": ["Trekking E-Bike 3500 Euro Test", "Trekking E-Bike Riemen 3500 Euro"],
    "research": {"category": "trekking", "use_case": "Pendeln", "electric": True, "extra_topic": None},
    "topic": "fahrrad",
}  # fmt: skip


@pytest.fixture(autouse=True)
def tavily(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(get_settings(), "tavily_api_key", "tvly-test")
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{TAVILY_URL}/search").respond(json=SEARCH_OK)
        yield router


async def run(message: str, recommender_model: Any, intake: dict[str, Any] | None = None) -> tuple[RunOutcome, list]:
    outcome = RunOutcome()
    async with httpx.AsyncClient() as http:
        deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute")
        with (
            intake_agent.override(model=TestModel(custom_output_args=intake or INTAKE)),
            recommender_agent.override(model=recommender_model),
        ):
            events = [event async for event in run_recommender(message, [], deps, outcome)]
    return outcome, events


def test_route_intake() -> None:
    assert route_intake({"prefetch": Prefetch()}) == ["recommender"]
    assert route_intake({"prefetch": Prefetch(off_topic="fachfremd")}) == ["reject_off_topic"]
    prefetch = Prefetch(
        searches=[SearchRequest(query="x")], research=ResearchRequest(category=BikeCategory.gravel, use_case="x")
    )
    assert route_intake({"prefetch": prefetch}) == ["prefetch_web_search", "prefetch_research"]


async def test_follow_up_skips_prefetch_searches_but_ranking_stays_due() -> None:
    """Beratung 21: Bei jeder Folgenachricht liefen dieselben allgemeinen Vorab-Suchen wie bei der ersten."""
    earlier = [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "Trekking-E-Bike bis 3500 €"}]},
        {
            "kind": "response",
            "parts": [{"part_kind": "tool-call", "tool_name": "search_bikes", "args": {"query": "a"}}],
        },
    ]
    async with httpx.AsyncClient() as http:
        deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute")
        with intake_agent.override(model=TestModel(custom_output_args=INTAKE)):
            first = await plan_prefetch("Trekking-E-Bike bis 3500 €", [], deps)
            follow_up = await plan_prefetch("suche jetzt", earlier, deps)
    assert first.ranking_due and len(first.searches) == 2
    assert follow_up.ranking_due and follow_up.searches == []


def test_prefetched_calls_become_history() -> None:
    prefetched = [
        {"tool_call_id": "pre_2", "tool_name": "research_knowledge", "args": {}, "result": {"error": "x"}},
        {"tool_call_id": "pre_1", "tool_name": "search_bikes", "args": {"query": "a"}, "result": {"results": []}},
    ]
    prompt, history = recommender_history({"message": "Gravelbike", "prefetched": prefetched})  # type: ignore[typeddict-item]
    assert prompt is None and history is not None and len(history) == 3
    # Websuche vor Recherche, unabhängig von der Reihenfolge der parallelen Knoten
    assert [p.tool_name for p in history[1].parts] == ["search_bikes", "research_knowledge"]


async def test_prefetch_runs_searches_and_research_before_recommender(tavily: Any) -> None:
    seen: list[list[str]] = []

    def respond(messages: list[ModelMessage]) -> Any:
        returns = [
            part.tool_name
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        seen.append(returns)
        if "recommend_frame_size" not in returns:
            return ("recommend_frame_size", {"category": "trekking", "height_cm": 175, "inseam_cm": 82})
        return ("final_recommendation", recommendation())

    outcome, events = await run("Trekking-E-Bike bis 3500 €", streaming_model(respond))
    assert outcome.error is None and outcome.recommendation is not None
    # Der Berater startet mit beiden Websuchen und der Recherche im Verlauf
    assert seen[0] == ["search_bikes", "search_bikes", "research_knowledge"]
    assert tavily.calls.call_count == 2
    tool_events = [e["tool"] for e in events if e["type"] == "tool_call"]
    assert tool_events.count("search_bikes") == 2 and "research_knowledge" in tool_events


async def test_off_topic_is_rejected_without_recommender() -> None:
    def fail(messages: list[ModelMessage]) -> Any:
        raise AssertionError("Berater darf nicht laufen")

    intake = INTAKE | {"topic": "fachfremd"}
    outcome, _ = await run("Wie backe ich Brot?", streaming_model(fail), intake)
    assert outcome.answer == OFF_TOPIC_MESSAGE and outcome.history_json is None


async def test_failing_intake_falls_back_to_recommender() -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Intake kaputt")

    answer = ("answer_user", {"message": "Ein Gravelbike ist ein Rennrad für Schotter."})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.agent.graph.plan_prefetch", broken)
        outcome, _ = await run("Was ist ein Gravelbike?", streaming_model(lambda _: answer))
    assert outcome.answer == "Ein Gravelbike ist ein Rennrad für Schotter."
