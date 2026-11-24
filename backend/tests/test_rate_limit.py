"""Drosselung auf Anfragen und Tokens pro Minute und Wiederholung nach einem 429 (app.agent.rate_limit)."""

from collections.abc import AsyncIterator

import pytest
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.exceptions import ModelRateLimitError
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.agent.models import MAX_OUTPUT_TOKENS, build_model
from app.agent.rate_limit import RATE_LIMIT_RETRIES, RateLimitedModel, RequestBudget, shared_budget


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_token_budget_waits_until_tokens_leave_window() -> None:
    clock = FakeClock()
    budget = RequestBudget(0, tokens_per_minute=100, clock=clock, sleep=clock.sleep)
    assert await budget.acquire(60) == 0
    clock.now = 10
    assert await budget.acquire(30) == 0  # 90 von 100
    clock.now = 20
    # 90 + 60 > 100: warten, bis die erste Anfrage (t=0) das Fenster verlässt
    assert await budget.acquire(60) == 41
    assert clock.now == 61


async def test_oversized_request_is_sent_into_empty_window() -> None:
    clock = FakeClock()
    budget = RequestBudget(0, tokens_per_minute=100, clock=clock, sleep=clock.sleep)
    assert await budget.acquire(500) == 0  # sonst wartete sie ewig
    assert await budget.acquire(1) == 61  # danach erst, wenn sie aus dem Fenster ist


async def test_requests_and_tokens_both_limit() -> None:
    clock = FakeClock()
    budget = RequestBudget(2, tokens_per_minute=1000, clock=clock, sleep=clock.sleep)
    await budget.acquire(10)
    await budget.acquire(10)
    assert await budget.acquire(10) == 61  # Anfragelimit greift trotz freier Tokens


def test_build_model_with_token_limit_only() -> None:
    model = build_model("test", tokens_per_minute=100_000)
    assert isinstance(model, RateLimitedModel) and model.budget is shared_budget("test", 0, 100_000)
    assert build_model("test") == "test"
    assert MAX_OUTPUT_TOKENS == 8_000


def flaky(fail_times: int, retry_after: float | None = 7.0) -> tuple[FunctionModel, list[int]]:
    """Modell, das zuerst `fail_times` mal mit 429 ablehnt und dann antwortet."""
    calls: list[int] = []

    def check() -> None:
        calls.append(1)
        if len(calls) <= fail_times:
            raise ModelRateLimitError("gpt-test", "Rate limit reached (TPM)", retry_after=retry_after)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        check()
        return ModelResponse(parts=[TextPart("ok")])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        check()
        yield "ok"

    return FunctionModel(respond, stream_function=stream), calls


def limited(model: FunctionModel, clock: FakeClock) -> RateLimitedModel:
    return RateLimitedModel(model, 0, budget=RequestBudget(0, 100_000, clock=clock, sleep=clock.sleep))


async def test_retries_after_rate_limit_with_provider_wait() -> None:
    clock = FakeClock()
    model, calls = flaky(fail_times=2)
    result = await Agent(limited(model, clock)).run("Hallo")
    assert result.output == "ok" and len(calls) == 3
    assert clock.slept == [8.0, 8.0]  # retry_after 7 s + 1 s Abstand


async def test_streaming_retries_when_opening_fails() -> None:
    clock = FakeClock()
    model, calls = flaky(fail_times=1, retry_after=None)
    async with Agent(limited(model, clock)).run_stream("Hallo") as stream:
        assert await stream.get_output() == "ok"
    assert len(calls) == 2 and clock.slept == [21.0]  # ohne Angabe 20 s + 1 s


async def test_gives_up_after_retries() -> None:
    clock = FakeClock()
    model, calls = flaky(fail_times=RATE_LIMIT_RETRIES + 1)
    with pytest.raises(ModelRateLimitError):
        await Agent(limited(model, clock)).run("Hallo")
    assert len(calls) == RATE_LIMIT_RETRIES + 1
