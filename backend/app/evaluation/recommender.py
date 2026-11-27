"""Misst Aufwand und Ergebnisqualität des Fahrradberaters mit echten Modellen und echter Websuche.

Jede Anfrage aus recommender_cases.json läuft wie im Chat durch den LangGraph-Graphen (Intake, Vorab-Phase, Berater).
Gemessen werden Laufzeit, Tool-Aufrufe (davon Tavily), Modellaufrufe, Tokens, Kosten des Beraters und
Wiederholungen durch die Selbstprüfung; geprüft werden Ausgabetyp, Länge der Rangliste, belegte Quell-URLs,
Budget, Kategorie, Motor und Rahmengröße.

Aufruf: python -m app.evaluation.recommender [--case ID] [--repeat N] [--output bericht.json]
Kostet echte Modell- und Tavily-Aufrufe und belegt dasselbe Ratenlimit wie die App.
"""

import argparse
import asyncio
import contextlib
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic_ai import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from app.agent.deps import AgentDeps
from app.agent.recommendation_schema import BikeRecommendation, ClarificationQuestion
from app.agent.recommender import BUDGET_TOLERANCE, known_pages, match_url
from app.agent.runner import RunOutcome, run_recommender
from app.config import get_settings
from app.db import SessionLocal, engine

CASES_FILE = Path(__file__).parent / "recommender_cases.json"
WEB_TOOLS = {"search_bikes", "read_product_page"}


@dataclass
class CaseRun:
    case: str
    seconds: float = 0.0
    output: str = "none"
    error: str | None = None
    tool_calls: dict[str, int] = field(default_factory=dict)
    web_calls: int = 0
    model_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    bikes: int = 0
    checks: dict[str, bool] = field(default_factory=dict)
    # Ausgegebene Rangliste bzw. Rückfrage, zum Nachvollziehen nicht bestandener Prüfungen
    result: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def check_output(case: dict[str, Any], outcome: RunOutcome, messages: list[Any]) -> dict[str, bool]:
    expected = case["expected_output"]
    if expected == "ask_user":
        return {"Rückfrage": outcome.question is not None}
    rec = outcome.recommendation
    if rec is None:
        return {"Rangliste": False}
    pages = known_pages(messages)
    checks = {
        "Rangliste": True,
        "1-5 Räder": 1 <= len(rec.ranking) <= 5,
        "Quellen belegt": all(match_url(b.price_source_url, pages) for b in rec.ranking),
        "Kategorie": all(b.category.value in case.get("category", [b.category.value]) for b in rec.ranking),
    }
    if budget := case.get("budget_eur"):
        checks["Budget"] = all(
            b.price_eur is None or b.price_eur <= budget * (1 + BUDGET_TOLERANCE) for b in rec.ranking
        )
    if case.get("electric") is False:
        checks["ohne Motor"] = all(not b.key_specs.motor for b in rec.ranking)
    if case.get("electric") is True:
        checks["mit Motor"] = all(b.key_specs.motor or b.key_specs.battery_wh for b in rec.ranking)
    if case.get("frame_size"):
        checks["Rahmengröße"] = all(b.frame_size for b in rec.ranking)
    return checks


def measure(run: CaseRun, messages: list[Any]) -> None:
    for message in messages:
        if isinstance(message, ModelResponse):
            run.model_requests += 1
            run.input_tokens += message.usage.input_tokens
            run.output_tokens += message.usage.output_tokens
            with contextlib.suppress(Exception):  # Modell ohne Preisangabe
                run.cost_usd += float(message.cost().total_price)
            for part in message.parts:
                if isinstance(part, ToolCallPart) and not part.tool_name.startswith(("final_", "ask_", "answer_")):
                    run.tool_calls[part.tool_name] = run.tool_calls.get(part.tool_name, 0) + 1
        elif isinstance(message, ModelRequest):
            run.retries += sum(isinstance(part, ToolReturnPart) and part.outcome == "retried" for part in message.parts)
    run.web_calls = sum(count for name, count in run.tool_calls.items() if name in WEB_TOOLS)


async def run_case(case: dict[str, Any], http: httpx.AsyncClient) -> CaseRun:
    run = CaseRun(case=case["id"])
    deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Evaluation")
    outcome = RunOutcome()
    started = time.perf_counter()
    async for _ in run_recommender(case["message"], [], deps, outcome):
        pass
    run.seconds = round(time.perf_counter() - started, 1)
    run.error = outcome.error
    messages = ModelMessagesTypeAdapter.validate_python(outcome.history_json or [])
    measure(run, messages)
    run.cost_usd = round(run.cost_usd, 4)
    if isinstance(outcome.recommendation, BikeRecommendation):
        run.output, run.bikes = "final_recommendation", len(outcome.recommendation.ranking)
        run.result = outcome.recommendation.model_dump(mode="json")
    elif isinstance(outcome.question, ClarificationQuestion):
        run.output = "ask_user"
        run.result = outcome.question.model_dump(mode="json")
    elif outcome.answer:
        run.output = "answer_user"
        run.result = {"message": outcome.answer}
    run.checks = {"Ausgabetyp": run.output == case["expected_output"]} | check_output(case, outcome, messages)
    return run


def summarize(runs: list[CaseRun]) -> dict[str, Any]:
    def mean(attr: str) -> float:
        return round(statistics.mean(getattr(r, attr) for r in runs), 2) if runs else 0.0

    checks = [ok for r in runs for ok in r.checks.values()]
    return {
        "runs": len(runs),
        "passed_runs": sum(r.passed for r in runs),
        "checks_passed": f"{sum(checks)}/{len(checks)}",
        "mean_seconds": mean("seconds"),
        "mean_web_calls": mean("web_calls"),
        "mean_model_requests": mean("model_requests"),
        "mean_input_tokens": mean("input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_retries": mean("retries"),
        "total_cost_usd": round(sum(r.cost_usd for r in runs), 3),
    }


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", help="nur diesen Fall")
    parser.add_argument("--repeat", type=int, default=1, help="Läufe je Fall")
    parser.add_argument("--output", type=Path, help="Bericht als JSON speichern")
    args = parser.parse_args(argv)
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"Fall {args.case} nicht gefunden")
            return 1
    runs: list[CaseRun] = []
    async with httpx.AsyncClient(timeout=get_settings().http_timeout_seconds) as http:
        for case in cases:
            for number in range(1, args.repeat + 1):
                run = await run_case(case, http)
                runs.append(run)
                failed = [name for name, ok in run.checks.items() if not ok]
                print(
                    f"{run.case} #{number}: {run.output}, {run.bikes} Räder, {run.seconds} s, "
                    f"{run.web_calls} Web-Aufrufe, {run.retries} Wiederholungen, {run.cost_usd:.4f} $ – "
                    + ("bestanden" if run.passed else f"NICHT bestanden: {', '.join(failed)}")
                    + (f" ({run.error})" if run.error else "")
                )
                if run.output == "answer_user" and run.result:
                    print(f"  Antwort: {run.result['message'][:400]}")
    await engine.dispose()
    summary = summarize(runs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        report = {"summary": summary, "runs": [asdict(r) | {"passed": r.passed} for r in runs]}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r.passed for r in runs) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
