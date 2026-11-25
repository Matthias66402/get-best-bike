"""Drosselt Anfragen an das Sprachmodell auf ein Limit an Anfragen und Tokens pro Minute.

Manche API-Keys erlauben nur wenige Anfragen bzw. Tokens pro Minute (Projektlimit, bei diesem Projekt für gpt-5-mini
5 Anfragen und 100.000 Tokens). Beim Streaming meldet OpenAI ein überschrittenes Limit mitunter nicht als HTTP 429,
sondern als Fehler im bereits geöffneten Stream; dann kann niemand die Anfrage wiederholen. Deshalb wird vor dem
Senden gewartet, statt das Limit zu reißen. Kommt trotzdem ein 429 beim Öffnen der Anfrage (z. B. weil die
Schätzung zu niedrig war oder ein anderer Prozess dasselbe Limit nutzt), wartet das Modell die vom Anbieter
genannte Zeit ab und versucht es erneut.

OpenAI rechnet beim Token-Limit mit den geschätzten Eingabe-Tokens plus dem angefragten Höchstwert für die Ausgabe
(`max_tokens`), nicht mit dem späteren Verbrauch. Die Schätzung hier macht es ebenso.

Das gilt je Prozess: Laufen App und Evaluation gleichzeitig, teilen sie sich das Limit trotzdem.
OpenAI zählt das Limit je Modell; Agenten mit demselben Modell teilen sich deshalb ein Budget (shared_budget).
"""

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from pydantic_ai import ModelMessage, ModelRequest, ModelResponse, RunContext, ThinkingPart, ToolCallPart
from pydantic_ai.exceptions import ModelRateLimitError
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from app.observability import add_metadata

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0
# Abstand zur Fenstergrenze, damit Uhrabweichungen zum Server nicht doch zu einem 429 führen
MARGIN_SECONDS = 1.0
# Zeichen je Token für den geschätzten Teil der Eingabe (Inhalte ohne JSON-Gerüst); gemessen an 330 Aufrufen von
# gpt-5-mini: Median 2,7, 95 % über 1,95. Bewusst niedrig, geschätzt wird nur der neue Teil (siehe estimate_tokens)
CHARS_PER_TOKEN = 2.0
# Zuschlag auf die Schätzung, z. B. für geänderte Anweisungen oder Tool-Schemas seit der letzten Antwort
ESTIMATE_MARGIN = 1.1
# Wiederholungen nach einem 429 beim Öffnen der Anfrage und Wartezeit, wenn der Anbieter keine nennt
RATE_LIMIT_RETRIES = 3
DEFAULT_RETRY_SECONDS = 20.0


class RequestBudget:
    """Gleitendes Fenster: höchstens `per_minute` Anfragen und `tokens_per_minute` Tokens in jeden 60 Sekunden.

    Ein Token-Bucket (Schub, danach eine Anfrage je 60/`per_minute` s) reichte nicht: OpenAI lehnte damit
    gelegentlich ab, zählt also mindestens so streng wie dieses Fenster. `tokens_per_minute` = 0 heißt ohne
    Token-Limit.
    """

    def __init__(
        self,
        per_minute: int,
        tokens_per_minute: int = 0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.per_minute = per_minute
        self.tokens_per_minute = tokens_per_minute
        self.clock = clock
        self.sleep = sleep
        # (Zeitpunkt, geschätzte Tokens) je gesendeter Anfrage
        self.sent: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    def _fits(self, tokens: int) -> bool:
        if self.per_minute > 0 and len(self.sent) >= self.per_minute:
            return False
        if self.tokens_per_minute <= 0 or not self.sent:
            return True  # Eine einzelne Anfrage über dem Limit muss trotzdem gesendet werden
        return sum(t for _, t in self.sent) + tokens <= self.tokens_per_minute

    async def acquire(self, tokens: int = 0) -> float:
        """Wartet, bis eine Anfrage mit `tokens` geschätzten Tokens erlaubt ist; gibt die Wartezeit zurück."""
        async with self._lock:
            waited = 0.0
            while True:
                now = self.clock()
                while self.sent and now - self.sent[0][0] >= WINDOW_SECONDS:
                    self.sent.popleft()
                if self._fits(tokens):
                    break
                # Bis die älteste Anfrage aus dem Fenster fällt; danach erneut prüfen
                wait = WINDOW_SECONDS - (now - self.sent[0][0]) + MARGIN_SECONDS
                await self.sleep(wait)
                waited += wait
            self.sent.append((self.clock(), tokens))
            return waited


_shared_budgets: dict[str, RequestBudget] = {}


def shared_budget(model: str, per_minute: int, tokens_per_minute: int = 0) -> RequestBudget:
    """Ein Budget je Modellname und Prozess, z. B. für Recherche-Agent und Relevanzprüfung mit demselben Modell."""
    budget = _shared_budgets.get(model)
    if budget is None or (budget.per_minute, budget.tokens_per_minute) != (per_minute, tokens_per_minute):
        budget = _shared_budgets[model] = RequestBudget(per_minute, tokens_per_minute)
    return budget


def content_chars(messages: list[ModelMessage]) -> int:
    """Zeichen, die als Eingabe beim Modell ankommen: Prompts, Texte, Tool-Argumente und -Ergebnisse, Anweisungen.

    Nicht das ganze Verlaufs-JSON: Zeitstempel, Feldnamen und Anbieter-Metadaten zählt der Anbieter nicht, sie
    machten die Schätzung drei- bis vierfach zu hoch (Beratung 22: 100.000 geschätzt, 29.000 tatsächlich).
    Reasoning-Zusammenfassungen gehen nicht als Eingabe zurück.
    """
    chars = 0
    for message in messages:
        if isinstance(message, ModelRequest) and message.instructions:
            chars += len(message.instructions)
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                chars += len(part.tool_name) + len(part.args_as_json_str())
            elif isinstance(part, ThinkingPart):
                continue
            elif isinstance(content := getattr(part, "content", None), str):
                chars += len(content)
            elif content is not None:
                chars += len(json.dumps(content, ensure_ascii=False, default=str))
    return chars


def estimate_tokens(
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> int:
    """Eingabe-Tokens der Anfrage plus angefragter Höchstwert für die Ausgabe.

    Selbstkalibrierend: Die letzte Antwort im Verlauf nennt den exakten Verbrauch der Anfrage davor (Eingabe plus
    Ausgabe, die nun im Verlauf steht); geschätzt wird nur, was seitdem dazukam (meist Tool-Ergebnisse), nach Zeichen.
    Ohne frühere Antwort zählen alle Inhalte und die Tool-Schemas. Der Zuschlag deckt Änderungen der Anweisungen ab.
    """
    last = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(message := messages[index], ModelResponse) and message.usage.input_tokens
        ),
        None,
    )
    if last is None:
        chars = content_chars(messages)
        for tool in [*model_request_parameters.function_tools, *model_request_parameters.output_tools]:
            chars += len(tool.name) + len(tool.description or "") + len(json.dumps(tool.parameters_json_schema))
        tokens = chars / CHARS_PER_TOKEN
    else:
        usage = messages[last].usage
        tokens = usage.input_tokens + usage.output_tokens + content_chars(messages[last + 1 :]) / CHARS_PER_TOKEN
    max_output = int((model_settings or {}).get("max_tokens") or 0)
    return int(tokens * ESTIMATE_MARGIN) + max_output


class RateLimitedModel(WrapperModel):
    """Modell, das jede Anfrage erst sendet, wenn das Budget es erlaubt, und nach einem 429 erneut versucht."""

    def __init__(
        self, wrapped: Model | str, per_minute: int, budget: RequestBudget | None = None, tokens_per_minute: int = 0
    ) -> None:
        super().__init__(wrapped)
        self.budget = budget or RequestBudget(per_minute, tokens_per_minute)

    async def _wait_after_rate_limit(self, exc: ModelRateLimitError, attempt: int) -> float:
        seconds = (exc.retry_after if exc.retry_after else DEFAULT_RETRY_SECONDS) + MARGIN_SECONDS
        logger.warning(
            "Anfragelimit von %s, Versuch %d/%d in %.0f s: %s",
            self.model_name, attempt + 1, RATE_LIMIT_RETRIES, seconds, exc,
        )  # fmt: skip
        await self.budget.sleep(seconds)
        return seconds

    @staticmethod
    def _report(tokens: int, waited: float) -> None:
        """Im Tracing an der Generation: Die Wartezeit steckt sonst unsichtbar in ihrer Dauer (Beratung 23: fast
        jeder Aufruf rund 60 s, davon der Großteil Warten auf das Minutenfenster)."""
        add_metadata(rate_limit_wait_s=round(waited, 1), rate_limit_estimated_tokens=tokens)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        tokens = estimate_tokens(messages, model_settings, model_request_parameters)
        waited = 0.0
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            waited += await self.budget.acquire(tokens)
            self._report(tokens, waited)
            try:
                return await super().request(messages, model_settings, model_request_parameters)
            except ModelRateLimitError as exc:
                if attempt == RATE_LIMIT_RETRIES:
                    raise
                waited += await self._wait_after_rate_limit(exc, attempt)
        raise AssertionError("unerreichbar")

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        tokens = estimate_tokens(messages, model_settings, model_request_parameters)
        async with AsyncExitStack() as stack:
            waited = 0.0
            for attempt in range(RATE_LIMIT_RETRIES + 1):
                waited += await self.budget.acquire(tokens)
                self._report(tokens, waited)
                try:
                    # Nur das Öffnen wird wiederholt; ein Fehler mitten im Stream geht an den Aufrufer
                    stream = await stack.enter_async_context(
                        super().request_stream(messages, model_settings, model_request_parameters, run_context)
                    )
                    break
                except ModelRateLimitError as exc:
                    if attempt == RATE_LIMIT_RETRIES:
                        raise
                    waited += await self._wait_after_rate_limit(exc, attempt)
            yield stream
