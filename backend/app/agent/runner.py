"""Führt einen Chat-Durchlauf über den LangGraph-Graphen (app.agent.graph) aus und streamt UI-taugliche Events."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.agent.deps import AgentDeps
from app.agent.graph import USAGE_LIMITS, RecommenderState, recommender_graph
from app.agent.recommendation_schema import BikeRecommendation, ClarificationQuestion

__all__ = ["USAGE_LIMITS", "RunOutcome", "run_recommender"]


@dataclass
class RunOutcome:
    """Wird während des Streams befüllt und danach vom Aufrufer gespeichert."""

    recommendation: BikeRecommendation | None = None
    question: ClarificationQuestion | None = None
    # Antwort ohne Rangliste
    answer: str | None = None
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    history_json: list[Any] | None = None
    error: str | None = None


def _apply_state(state: RecommenderState, outcome: RunOutcome) -> None:
    output = state.get("output")
    if isinstance(output, BikeRecommendation):
        outcome.recommendation = output
    elif isinstance(output, ClarificationQuestion):
        outcome.question = output
    outcome.answer = state.get("answer")
    outcome.history_json = state.get("new_history_json")
    outcome.error = state.get("error")


async def run_recommender(
    message: str, history_json: list[Any], deps: AgentDeps, outcome: RunOutcome
) -> AsyncIterator[dict[str, Any]]:
    """Streamt Events: token, tool_call, tool_result, error.

    Text und Tool-Aufrufe landen schon während des Streams im `outcome`, damit sie auch bei einem Abbruch durch den
    Client erhalten bleiben; Ergebnis und Verlauf erst nach dem letzten Graph-Schritt.
    """
    calls_by_id: dict[str, dict[str, Any]] = {}
    final: RecommenderState | None = None
    graph_input: RecommenderState = {"message": message, "history_json": history_json}
    try:
        async for mode, chunk in recommender_graph.astream(graph_input, context=deps, stream_mode=["custom", "values"]):
            if mode == "values":
                final = chunk
                continue
            event: dict[str, Any] = chunk
            if event["type"] == "token":
                outcome.text += event["content"]
            elif event["type"] == "tool_call":
                call = {key: value for key, value in event.items() if key != "type"}
                calls_by_id[call["id"]] = call
                outcome.tool_calls.append(call)
            elif event["type"] == "tool_result" and (call := calls_by_id.get(event["id"])):
                call["result"] = event["result"]
                call["failed"] = event["failed"]
            yield event
    except Exception as exc:  # Fehler im Graphen selbst; Fehler des Agenten fängt schon der Berater-Knoten ab
        outcome.error = f"Fehler bei der Beratung: {type(exc).__name__}: {exc}"
    else:
        if final is not None:
            _apply_state(final, outcome)

    if outcome.error:
        yield {"type": "error", "message": outcome.error}
