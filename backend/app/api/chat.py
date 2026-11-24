import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic_ai import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

from app.agent.deps import AgentDeps
from app.agent.runner import RunOutcome, run_recommender
from app.api.consultations import get_owned_consultation
from app.auth.deps import CurrentUser, SessionDep
from app.config import get_settings
from app.db import SessionLocal
from app.models import ChatMessage, Consultation, ConsultationStatus
from app.observability import chat_turn
from app.rag.attribution import retrieved_documents, web_sources
from app.schemas.consultation import ChatRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


def sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def build_stored_recommendation(outcome: RunOutcome, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Rangliste plus Quellen, die das Backend aus dem Verlauf bestimmt (nicht aus den `sources` des LLM).

    `web_sources`: die Quell-URLs der Rangliste mit Seitentitel; `knowledge_sources`: Titel der Ratgeber-Dokumente.
    """
    assert outcome.recommendation is not None
    urls = [bike.price_source_url for bike in outcome.recommendation.ranking]
    return outcome.recommendation.model_dump(mode="json") | {
        "web_sources": web_sources(history, urls),
        "knowledge_sources": retrieved_documents(history),
    }


def assistant_text(outcome: RunOutcome) -> str:
    if outcome.question:
        return outcome.question.question
    if outcome.answer:
        return outcome.answer
    if outcome.recommendation:
        return (
            outcome.text.strip()
            or f"Hier ist Deine Rangliste „{outcome.recommendation.title}“. {outcome.recommendation.summary}"
        )
    return outcome.error or outcome.text.strip() or "Keine Antwort erhalten."


def outcome_kind(outcome: RunOutcome) -> str:
    """Art des Ergebnisses, z. B. zum Filtern der Traces in Langfuse."""
    if outcome.error:
        return "error"
    if outcome.recommendation:
        return "recommendation"
    return "question" if outcome.question else "answer" if outcome.answer else "none"


def with_unanswered_message(history: list[Any], user_message: str) -> list[Any]:
    """Verlauf plus die Nachricht eines gescheiterten Laufs (z. B. Netzwerkfehler), damit sie dem Berater beim
    nächsten Lauf nicht fehlt. PydanticAI hängt die nächste Nachricht an diese offene Anfrage an."""
    request = ModelRequest(parts=[UserPromptPart(user_message)])
    return [*history, *ModelMessagesTypeAdapter.dump_python([request], mode="json")]


async def save_outcome(consultation_id: int, user_message: str, outcome: RunOutcome) -> dict[str, Any] | None:
    """Speichert Nachrichten, Verlauf und ggf. Rangliste; gibt die gespeicherte Rangliste zurück."""
    async with SessionLocal() as session:
        consultation = await session.get(Consultation, consultation_id)
        if consultation is None:  # während des Laufs gelöscht
            return None
        session.add(ChatMessage(consultation_id=consultation_id, role="user", content=user_message, tool_calls=[]))
        session.add(
            ChatMessage(
                consultation_id=consultation_id,
                role="assistant",
                content=assistant_text(outcome),
                tool_calls=outcome.tool_calls,
            )
        )
        if outcome.history_json is not None:
            consultation.agent_history = outcome.history_json
        elif outcome.error:
            consultation.agent_history = with_unanswered_message(consultation.agent_history or [], user_message)
        stored = None
        if outcome.recommendation is not None:
            stored = build_stored_recommendation(outcome, consultation.agent_history or [])
            consultation.recommendation = stored
            consultation.title = outcome.recommendation.title
            consultation.status = ConsultationStatus.recommended
        await session.commit()
        return stored


@router.post("/{consultation_id}")
async def chat(
    consultation_id: int, data: ChatRequest, user: CurrentUser, session: SessionDep, request: Request
) -> StreamingResponse:
    """Schickt eine Nachricht an den Fahrradberater und streamt dessen Arbeit als Server-Sent Events.

    Events: `token`, `tool_call`, `tool_result`, `question`, `answer`, `recommendation`, `error`, `done`.
    """
    consultation = await get_owned_consultation(session, consultation_id, user.id)
    history = list(consultation.agent_history or [])
    user_name, user_id = user.display_name, user.id
    height_cm, inseam_cm = user.height_cm, user.inseam_cm
    # Die Request-Session wird nicht in den Stream mitgenommen; Tools und Speichern öffnen eigene Sessions.
    await session.close()

    async def stream() -> AsyncIterator[str]:
        settings = get_settings()
        outcome = RunOutcome()
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
            deps = AgentDeps(
                settings=settings, http=http, session_factory=SessionLocal, user_name=user_name,
                height_cm=height_cm, inseam_cm=inseam_cm,
            )  # fmt: skip
            with chat_turn(message=data.message, user_id=user_id, consultation_id=consultation_id) as trace:
                async with aclosing(run_recommender(data.message, history, deps, outcome)) as events:
                    async for event in events:
                        if await request.is_disconnected():
                            break
                        yield sse(event)
                trace.update(
                    output=assistant_text(outcome),
                    metadata={"outcome": outcome_kind(outcome), "tool_calls": len(outcome.tool_calls)},
                    **({"level": "ERROR", "status_message": outcome.error} if outcome.error else {}),
                )

        stored = await save_outcome(consultation_id, data.message, outcome)
        if stored is not None:
            yield sse({"type": "recommendation", "recommendation": stored})
        elif outcome.question is not None:
            yield sse({"type": "question", **outcome.question.model_dump()})
        elif outcome.answer is not None:
            yield sse({"type": "answer", "content": outcome.answer})
        yield sse({"type": "done"})

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
