import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"

# Provider-SDKs (OpenAI, Anthropic, ...) lesen ihre API-Keys aus os.environ.
load_dotenv(ENV_FILE)
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str = "postgresql+asyncpg://bike:bike@localhost:5441/bike"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 12 * 60

    admin_email: str = "admin@example.com"
    admin_password: str = "admin12345"

    llm_model: str = "openai:gpt-5-mini"
    # Reasoning-Aufwand des Planers (minimal, low, medium, high); leer = Standard des Modells
    llm_reasoning_effort: str = ""
    # Anfragelimit des API-Keys je Modell (Anfragen pro Minute) für alle Agenten; 0 = nicht drosseln
    llm_requests_per_minute: int = 0
    # Token-Limit des API-Keys je Modell (Tokens pro Minute, Eingabe + max. Ausgabe); 0 = nicht drosseln
    llm_tokens_per_minute: int = 0
    embedding_model: str = "openai:text-embedding-3-small"
    embedding_dim: int = 1536
    # Bewertet Suchtreffer der Wissensbasis vor der Weitergabe an den Agenten; leer = nur Mindestähnlichkeit.
    # Anderes Modell als der Recherche-Agent: Eine Recherche löst je Suche eine Bewertung aus; mit gemeinsamem
    # Anfragelimit (je Modell) musste sie sonst oft auf das nächste Minutenfenster warten.
    relevance_model: str = "openai:gpt-4o-mini"
    # Recherche-Agent, an den der Planer die Suche in der Wissensbasis delegiert (app.agent.research)
    research_model: str = "openai:gpt-4.1-mini"
    # Bestimmt vorab Kategorie, Budget, Körpermaße und Suchanfragen (app.agent.intake); leer = aus
    intake_model: str = "openai:gpt-4o-mini"

    # Websuche über Tavily (app.agent.tools.web); leer = Websuche aus
    tavily_api_key: str = ""
    tavily_api_url: str = "https://api.tavily.com"
    web_search_max_results: int = 5
    # Suchergebnisse und Produktseiten so lange zwischenspeichern (Tabelle web_cache)
    web_cache_hours: int = 24
    http_timeout_seconds: float = 20.0

    cors_origins: str = "http://localhost:5181"

    # Tracing mit Langfuse (siehe app/observability.py); ohne Keys aus
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_tracing_enabled: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
