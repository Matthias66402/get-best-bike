import os

# Vor jedem App-Import auf die Test-Datenbank umstellen
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://bike:bike@localhost:5441/bike_test"
)
# Keine Traces aus Tests nach Langfuse senden, auch wenn .env Keys enthält
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
# Nie die echte Websuche: Tests, die sie brauchen, setzen einen Test-Key und mocken Tavily mit respx
os.environ["TAVILY_API_KEY"] = ""
# Ohne Drosselung: Tests ersetzen die Modelle ohnehin (Drosselung selbst: test_rate_limit)
os.environ["LLM_REQUESTS_PER_MINUTE"] = "0"
os.environ["LLM_TOKENS_PER_MINUTE"] = "0"

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from pydantic_ai.embeddings import TestEmbeddingModel
from pydantic_ai.models.test import TestModel
from sqlalchemy import text

from app.agent.intake import intake_agent
from app.agent.research import research_agent
from app.auth.security import hash_password
from app.config import get_settings
from app.db import SessionLocal, engine
from app.main import app
from app.models import User, UserRole
from app.rag.embeddings import get_embedder
from app.rag.relevance import get_relevance_agent

PASSWORD = "secret-pass-123"


@pytest.fixture(autouse=True)
async def clean_db() -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE users, consultations, chat_messages, documents, chunks, web_cache RESTART IDENTITY CASCADE")
        )
    yield


@pytest.fixture(scope="session", autouse=True)
async def dispose_engine() -> AsyncIterator[None]:
    yield
    await engine.dispose()


@pytest.fixture(autouse=True)
def fake_embeddings() -> Iterator[None]:
    with get_embedder().override(model=TestEmbeddingModel(dimensions=get_settings().embedding_dim)):
        yield


@pytest.fixture(autouse=True)
def keep_all_hits() -> Iterator[None]:
    """Relevanzprüfung ohne LLM: keine Bewertungen, also bleiben alle Treffer erhalten."""
    with get_relevance_agent().override(model=TestModel(custom_output_args={"judgements": []})):
        yield


@pytest.fixture(autouse=True)
def fake_research() -> Iterator[None]:
    """Recherche-Agent ohne LLM: TestModel ruft search_knowledge einmal auf und gibt eine generierte Zusammenfassung aus."""
    with research_agent.override(model=TestModel()):
        yield


@pytest.fixture(autouse=True)
def no_prefetch() -> Iterator[None]:
    """Intake ohne LLM und ohne Vorab-Suche: Der Berater läuft wie ohne Intake (Tests mit Vorab-Suche: test_graph)."""
    no_prefetch = {
        "category": None, "electric": None, "budget_eur": None, "use_case": None, "height_cm": None,
        "inseam_cm": None, "search_queries": [], "research": None, "topic": "fahrrad",
    }  # fmt: skip
    with intake_agent.override(model=TestModel(custom_output_args=no_prefetch)):
        yield


@pytest.fixture(autouse=True)
def short_rankings_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Geskriptete Tests geben meist 1-2 Räder aus; die Pflicht zu 5 Rädern prüft test_recommender gezielt."""
    monkeypatch.setattr("app.agent.recommender.MIN_RANKING_RETRIES", 0)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def create_user(email: str, role: UserRole = UserRole.user, name: str = "Test") -> User:
    async with SessionLocal() as session:
        user = User(email=email, display_name=name, password_hash=hash_password(PASSWORD), role=role)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def auth_headers(client: httpx.AsyncClient, email: str) -> dict[str, str]:
    response = await client.post("/api/auth/login", data={"username": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
async def user_headers(client: httpx.AsyncClient) -> dict[str, str]:
    await create_user("user@example.com", name="Ute")
    return await auth_headers(client, "user@example.com")


@pytest.fixture
async def admin_headers(client: httpx.AsyncClient) -> dict[str, str]:
    await create_user("admin@example.com", role=UserRole.admin, name="Admin")
    return await auth_headers(client, "admin@example.com")
