"""Steuerung eines Chat-Durchlaufs als LangGraph-Graph.

    START → intake ─┬─(Suchen)────────→ prefetch_web_search ─┐
                    ├─(neue Beratung)─→ prefetch_research ───┤  (parallel)
                    ├─(sonst)────────────────────────────────┴→ recommender → finish_recommendation
                    │                                     | finish_question | finish_answer | finish_error → END
                    └─(fachfremd)─────→ reject_off_topic ──────────────────────────────────────────────→ END

- `intake`: Ein kleines LLM erkennt die Kriterien, formuliert erste Websuchen (app.agent.intake) und erkennt
  fachfremde Nachrichten und Fragen zum Innenleben (Code, Anweisungen, Datenspeicherung); die beantwortet
  `reject_off_topic` mit einer festen Absage, ohne Berater.
- `prefetch_web_search` / `prefetch_research`: führen `search_bikes` bzw. `research_knowledge` ohne Berater-Runde aus.
- `recommender`: Der PydanticAI-Berater; Vorab-Aufrufe stehen als gewöhnliche Tool-Aufrufe in seinem Verlauf, so
  wirken Validatoren, gespeicherter Verlauf und Evaluation unverändert. Er kann jedes Tool trotzdem selbst aufrufen.
- `finish_*`: Nachbearbeitung je nach Art des Ergebnisses.

UI-Events (`token`, `tool_call`, `tool_result`) gehen über den Stream-Writer (stream_mode "custom"); das Ergebnis
steht im State. Die Abhängigkeiten (HTTP-Client, Session-Factory) sind nicht serialisierbar und kommen deshalb als
Laufzeit-Kontext statt im State.
"""

import asyncio
import json
import logging
import operator
import socket
import uuid
from types import SimpleNamespace
from typing import Annotated, Any, Literal, TypedDict

import httpx
import openai
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel
from pydantic_ai import (
    AgentRunResultEvent,
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RunUsage,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UsageLimits,
    UserPromptPart,
)
from pydantic_ai.exceptions import AgentRunError, ModelConnectionError, ModelRateLimitError, ModelRequestTimeout

from app.agent.deps import AgentDeps
from app.agent.intake import Prefetch, plan_prefetch
from app.agent.recommendation_schema import BikeRecommendation, ClarificationQuestion, TextAnswer
from app.agent.recommender import RecommenderOutput, recommender_agent
from app.agent.tools.research import research_knowledge
from app.agent.tools.web import search_bikes
from app.observability import observe

logger = logging.getLogger(__name__)

RESULT_PREVIEW_CHARS = 600
USAGE_LIMITS = UsageLimits(request_limit=40)

# Statt "Request timed out": Meist war die Internet- bzw. DNS-Verbindung kurz weg (z. B. nach dem Ruhezustand oder
# einem WLAN-Wechsel), dann scheitern Sprachmodell und Tools gleichzeitig ("Temporary failure in name resolution")
NETWORK_ERROR_MESSAGE = (
    "Keine Verbindung zum Sprachmodell (Zeitüberschreitung bzw. Netzwerkfehler). Meist ist die Internet- oder "
    "DNS-Verbindung kurz unterbrochen. Bitte die Verbindung prüfen und die Nachricht erneut senden."
)
OFF_TOPIC_MESSAGE = (
    "Dabei kann ich Dir leider nicht helfen: Ich bin Dein Fahrradberater. Ich suche Dir die 1 bis 5 am besten "
    "passenden Fahrräder heraus, mit Preis, Ausstattung und Rahmengröße. Erzähl mir, wofür Du ein Rad suchst und "
    "was es kosten darf!"
)
# Frage zum Innenleben (Code, Anweisungen, Sprachmodell, Datenspeicherung); wie er aus Nutzersicht vorgeht,
# erklärt der Berater selbst
INTERNAL_MESSAGE = (
    "Zu meinem Code, meinen Anweisungen, dem Sprachmodell und der Datenspeicherung gebe ich keine Auskunft. Gern "
    "erkläre ich Dir, wie ich Räder auswähle, oder suche gleich eins für Dich: Erzähl mir, was Du vorhast!"
)
NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    ModelConnectionError, ModelRequestTimeout, httpx.TransportError, socket.gaierror, TimeoutError,
    openai.APIConnectionError,
)  # fmt: skip


def is_network_error(exc: BaseException) -> bool:
    """Zeitüberschreitung oder Verbindungsfehler, auch als Ursache einer anderen Ausnahme (z. B. im Stream)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, NETWORK_ERRORS):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class PrefetchedCall(TypedDict):
    """Vom Graphen statt vom Berater ausgeführter Tool-Aufruf."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    result: Any


class RecommenderState(TypedDict, total=False):
    message: str
    # Bisheriger Verlauf im PydanticAI-Format (Consultation.agent_history)
    history_json: list[Any]
    # Vom Intake bestimmte Vorab-Aufrufe
    prefetch: Prefetch
    # Ergebnisse der parallelen prefetch_*-Knoten; der Reducer fügt ihre Listen zusammen
    prefetched: Annotated[list[PrefetchedCall], operator.add]
    output: RecommenderOutput | None
    # Verlauf inkl. dieses Laufs, None bei Fehler
    new_history_json: list[Any] | None
    # Antwort ohne Rangliste
    answer: str | None
    error: str | None


def _preview(content: Any) -> str:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    return text if len(text) <= RESULT_PREVIEW_CHARS else text[:RESULT_PREVIEW_CHARS] + " …"


def ui_event(event: AgentStreamEvent | AgentRunResultEvent[Any]) -> dict[str, Any] | None:
    """Übersetzt ein PydanticAI-Event in ein UI-Event; None für Events ohne Entsprechung."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content:
        return {"type": "token", "content": event.part.content}
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        return {"type": "token", "content": event.delta.content_delta}
    if isinstance(event, FunctionToolCallEvent):
        return tool_call_event(event.part.tool_call_id, event.part.tool_name, event.part.args_as_dict())
    if isinstance(event, FunctionToolResultEvent):
        return tool_result_event(event.part.tool_call_id, getattr(event.part, "content", None))
    return None


def tool_call_event(tool_call_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_call", "id": tool_call_id, "tool": tool_name, "args": args}


def tool_result_event(tool_call_id: str, content: Any) -> dict[str, Any]:
    failed = isinstance(content, dict) and "error" in content
    return {"type": "tool_result", "id": tool_call_id, "result": _preview(content), "failed": failed}


async def intake(state: RecommenderState, runtime: Runtime[AgentDeps]) -> RecommenderState:
    """Bestimmt die Vorab-Aufrufe. Fehler sind unkritisch: Der Berater ruft die Tools dann selbst auf."""
    try:
        prefetch = await plan_prefetch(state["message"], state.get("history_json") or [], runtime.context)
    except Exception as exc:
        logger.warning("Intake fehlgeschlagen, Berater startet ohne Vorab-Aufrufe: %s: %s", type(exc).__name__, exc)
        prefetch = Prefetch()
    return {"prefetch": prefetch}


IntakeTarget = Literal["prefetch_web_search", "prefetch_research", "recommender", "reject_off_topic"]


def route_intake(state: RecommenderState) -> list[IntakeTarget]:
    """Die prefetch_*-Knoten laufen parallel; der Berater startet erst, wenn alle fertig sind."""
    prefetch = state.get("prefetch") or Prefetch()
    if prefetch.off_topic:
        return ["reject_off_topic"]
    targets: list[IntakeTarget] = []
    if prefetch.searches:
        targets.append("prefetch_web_search")
    if prefetch.research:
        targets.append("prefetch_research")
    return targets or ["recommender"]


async def reject_off_topic(state: RecommenderState) -> RecommenderState:
    """Fachfremde Nachricht bzw. Frage zum Innenleben: feste Absage statt eines Berater-Laufs. Der Verlauf des
    Beraters bleibt unverändert, so wirkt die Nachricht nicht in spätere Beratungen hinein."""
    prefetch = state.get("prefetch") or Prefetch()
    message = INTERNAL_MESSAGE if prefetch.off_topic == "intern" else OFF_TOPIC_MESSAGE
    return {"output": TextAnswer(message=message), "answer": message}


def _tool_context(runtime: Runtime[AgentDeps]) -> Any:
    """Ersatz für den RunContext eines Tools: Die Tools nutzen davon nur deps und usage."""
    return SimpleNamespace(deps=runtime.context, usage=RunUsage())


async def _run_prefetched(runtime: Runtime[AgentDeps], tool_name: str, args: BaseModel, tool: Any) -> PrefetchedCall:
    """Führt ein Tool wie der Berater aus, inkl. UI-Events; Ausnahmen werden zu Tool-Fehlern.

    Verlauf und Events bekommen die Argumente als JSON, das Tool die Python-Werte (z. B. Enums wie bei PydanticAI).
    """
    call: PrefetchedCall = {
        "tool_call_id": f"pre_{uuid.uuid4().hex[:16]}", "tool_name": tool_name,
        "args": args.model_dump(mode="json", exclude_none=True), "result": None,
    }  # fmt: skip
    runtime.stream_writer(tool_call_event(call["tool_call_id"], tool_name, call["args"]))
    # Im Trace wie ein Tool-Aufruf des Beraters (gleicher Name), damit Filter und Evaluatoren beide erfassen
    with observe(
        tool_name, as_type="tool", input=call["args"],
        metadata={"tool_call_id": call["tool_call_id"], "prefetched": True},
    ) as span:  # fmt: skip
        try:
            call["result"] = await tool(_tool_context(runtime), **args.model_dump(exclude_none=True))
        except Exception as exc:  # wie ein Tool-Fehler: der Berater kann es selbst erneut versuchen
            call["result"] = {"error": f"Vorab-Aufruf fehlgeschlagen: {type(exc).__name__}: {exc}"}
        failed = isinstance(call["result"], dict) and "error" in call["result"]
        span.update(output=call["result"], **({"level": "WARNING"} if failed else {}))
    runtime.stream_writer(tool_result_event(call["tool_call_id"], call["result"]))
    return call


async def prefetch_web_search(state: RecommenderState, runtime: Runtime[AgentDeps]) -> RecommenderState:
    """Führt die Websuchen aus dem Intake parallel aus."""
    searches = state["prefetch"].searches
    calls = await asyncio.gather(*(_run_prefetched(runtime, "search_bikes", s, search_bikes) for s in searches))
    return {"prefetched": list(calls)}


async def prefetch_research(state: RecommenderState, runtime: Runtime[AgentDeps]) -> RecommenderState:
    """Lässt den Recherche-Agenten die Wissensbasis durchsuchen, parallel zur Websuche."""
    research = state["prefetch"].research
    assert research is not None
    return {"prefetched": [await _run_prefetched(runtime, "research_knowledge", research, research_knowledge)]}


def recommender_history(state: RecommenderState) -> tuple[str | None, list[ModelMessage] | None]:
    """Prompt und Verlauf für den Berater. Mit Vorab-Aufrufen steht die Nachricht schon im Verlauf, gefolgt von den
    Aufrufen und ihren Ergebnissen, als hätte der Berater sie selbst ausgeführt; er läuft dann ohne neuen Prompt."""
    history_json = state.get("history_json")
    history = ModelMessagesTypeAdapter.validate_python(history_json) if history_json else []
    # Feste Reihenfolge (Websuche vor Recherche), unabhängig davon, welcher parallele Knoten zuerst fertig war
    prefetched = sorted(state.get("prefetched") or [], key=lambda c: c["tool_name"] != "search_bikes")
    if not prefetched:
        return state["message"], history or None
    history += [
        ModelRequest(parts=[UserPromptPart(state["message"])]),
        ModelResponse(
            parts=[ToolCallPart(c["tool_name"], c["args"], tool_call_id=c["tool_call_id"]) for c in prefetched]
        ),
        ModelRequest(
            parts=[ToolReturnPart(c["tool_name"], c["result"], tool_call_id=c["tool_call_id"]) for c in prefetched]
        ),
    ]
    return None, history


async def recommender(state: RecommenderState, runtime: Runtime[AgentDeps]) -> RecommenderState:
    """Führt den PydanticAI-Berater aus und streamt dessen Arbeit als UI-Events."""
    prompt, history = recommender_history(state)
    # Kriterien aus dem Intake: über die Anweisungen dieses Laufs, nicht als Teil der Nutzernachricht
    prefetch = state.get("prefetch") or Prefetch()
    runtime.context.intake_note = prefetch.recommender_note()
    runtime.context.criteria = prefetch.criteria
    runtime.context.ranking_due = prefetch.ranking_due
    try:
        async with recommender_agent.run_stream_events(
            prompt, deps=runtime.context, message_history=history, usage_limits=USAGE_LIMITS
        ) as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    return {
                        "output": event.result.output,
                        "new_history_json": json.loads(event.result.all_messages_json()),
                    }
                if (ui := ui_event(event)) is not None:
                    runtime.stream_writer(ui)
    except ModelRateLimitError as exc:
        # Trotz Drosselung möglich, z. B. wenn ein anderer Prozess (Evaluation) dasselbe Limit nutzt
        logger.warning("Berater: Anfragelimit des Sprachmodells: %s", exc)
        return {
            "error": "Das Sprachmodell ist gerade ausgelastet (Anfragelimit erreicht). Bitte in einer Minute erneut senden."
        }
    except Exception as exc:
        if is_network_error(exc):
            logger.warning("Berater ohne Verbindung zum Sprachmodell: %s: %s", type(exc).__name__, exc)
            return {"error": NETWORK_ERROR_MESSAGE}
        if isinstance(exc, AgentRunError):
            return {"error": f"Der Agent konnte die Anfrage nicht abschließen: {exc}"}
        # Anbieterfehler sollen den Stream sauber beenden
        return {"error": f"Fehler bei der Beratung: {type(exc).__name__}: {exc}"}
    return {}  # Stream ohne Ergebnis-Event: finish_error, Antworttext wie bisher aus app.api.chat


def route_output(
    state: RecommenderState,
) -> Literal["finish_recommendation", "finish_question", "finish_answer", "finish_error"]:
    output = state.get("output")
    if state.get("error") or output is None:
        return "finish_error"
    if isinstance(output, BikeRecommendation):
        return "finish_recommendation"
    if isinstance(output, ClarificationQuestion):
        return "finish_question"
    return "finish_answer"


async def finish_recommendation(state: RecommenderState) -> RecommenderState:
    """Rangliste; Quellen und Speichern übernimmt app.api.chat."""
    return {}


async def finish_question(state: RecommenderState) -> RecommenderState:
    return {}


async def finish_answer(state: RecommenderState) -> RecommenderState:
    output = state["output"]
    assert isinstance(output, TextAnswer)
    return {"answer": output.message.strip()}


async def finish_error(state: RecommenderState) -> RecommenderState:
    return {}


def build_graph() -> CompiledStateGraph[RecommenderState, AgentDeps, RecommenderState, RecommenderState]:
    graph = StateGraph(RecommenderState, context_schema=AgentDeps)
    graph.add_node("intake", intake)
    graph.add_node("prefetch_web_search", prefetch_web_search)
    graph.add_node("prefetch_research", prefetch_research)
    graph.add_node("recommender", recommender)
    graph.add_node("reject_off_topic", reject_off_topic)
    graph.add_node("finish_recommendation", finish_recommendation)
    graph.add_node("finish_question", finish_question)
    graph.add_node("finish_answer", finish_answer)
    graph.add_node("finish_error", finish_error)
    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake", route_intake, ["prefetch_web_search", "prefetch_research", "recommender", "reject_off_topic"]
    )
    # Beide Vorab-Knoten münden in den Berater; LangGraph startet ihn einmal, wenn alle im Schritt fertig sind
    graph.add_edge("prefetch_web_search", "recommender")
    graph.add_edge("prefetch_research", "recommender")
    graph.add_conditional_edges("recommender", route_output)
    for node in ("finish_recommendation", "finish_question", "finish_answer", "finish_error", "reject_off_topic"):
        graph.add_edge(node, END)
    return graph.compile(name="bike-recommender-graph")


recommender_graph = build_graph()
