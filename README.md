# Get Best Bike – Fahrradberater mit Agent

Eingeloggte Nutzer beschreiben in eigenen Worten, welches Rad sie suchen, etwa „Trekking-E-Bike bis 3.500 € zum täglichen Pendeln, 15 km, teils Schotter, 1,75 m, Schrittlänge 82 cm“. Ein Berater-Agent entscheidet selbst, welche Tools er aufruft: Live-Websuche nach aktuellen Modellen und Preisen (Tavily), Produktseiten lesen, Rahmengröße berechnen und eine Wissensbasis mit Kaufratgebern (RAG, über einen eigenen Recherche-Agenten). Fehlen Angaben, fragt er nach. Das Ergebnis ist eine **Rangliste der 1 bis 5 am besten passenden Fahrräder**. Bevor sie angezeigt wird, prüft das Backend, ob Preise und Quellen belegt sind.

**Funktionen**

- **Beratung im Chat:** Der Agent arbeitet live sichtbar. Jeder Tool-Aufruf erscheint mit Argumenten und Ergebnis. Folgenachrichten wie „etwas günstiger“, „lieber ohne Motor“ oder „erklär mir Platz 2“ passen die Rangliste an bzw. beantworten die Frage.
- **Rangliste:** Platz 1 bis 5 mit Preis, Match-Score, Rahmengröße, wichtigsten Daten (Motor, Akku, Antrieb, Gewicht, Federung, Bremsen), Vor- und Nachteilen, Begründung und Link zum Angebot. Dazu eine Vergleichstabelle und Kauftipps.
- **Belegte Angaben:** Jede Quell-URL muss aus der Websuche dieser Beratung stammen. Jeder Preis muss im Text dieser Seite stehen, und die Marke muss auf der Seite vorkommen. Andernfalls geht die Rangliste mit konkretem Hinweis zurück an das Modell.
- **Rahmengröße:** deterministisch aus Körpergröße bzw. Schrittlänge, aus der Nachricht oder dem Profil. Faustformeln je Kategorie; eine Größentabelle des Herstellers geht vor.
- **Nachvollziehbare Quellen:** Webseiten und Ratgeber-Dokumente ermittelt das Backend aus dem Verlauf des Agenten, nicht aus Angaben des LLM.
- **Messbare Qualität:** Evaluationen für die Suchqualität der Wissensbasis und für Aufwand und Ergebnis des Beraters.

| Schicht | Technik |
|---|---|
| Frontend | React 19, TypeScript, Vite, React Router |
| Backend | FastAPI, SQLAlchemy 2 (async), Alembic, JWT-Auth |
| Agent | LangGraph (Ablauf eines Chat-Durchlaufs), PydanticAI (Berater, Intake und Recherche-Agent, anbieter-agnostisch, Modell per `.env`) |
| RAG | pgvector (HNSW, Kosinus), Embeddings über PydanticAI `Embedder`, Relevanzprüfung der Treffer durch ein kleines LLM |
| Websuche | [Tavily](https://tavily.com) (`/search` und `/extract`), Antworten 24 h in Postgres zwischengespeichert |
| Tracing | Langfuse oder LangSmith (optional, OpenTelemetry über die PydanticAI-Instrumentierung) |

## Schnellstart

Voraussetzungen: Docker und Node 20+.

```bash
cp .env.example .env            # OPENAI_API_KEY, TAVILY_API_KEY und JWT_SECRET eintragen
docker compose up -d --build    # Postgres/pgvector (Port 5441) + Backend (Port 8011, Hot-Reload)
docker compose exec backend alembic upgrade head
docker compose exec backend python -m app.seed      # Admin-Konto + Ratgeber-Dokumente einlesen (braucht OPENAI_API_KEY)

cd frontend && npm install && npm run dev            # http://localhost:5181
```

- Anmeldung mit `ADMIN_EMAIL` / `ADMIN_PASSWORD` aus der `.env`. Neue Nutzer können sich auch selbst registrieren.
- Einen kostenlosen Tavily-Key gibt es unter [app.tavily.com](https://app.tavily.com) (Free-Tier: 1.000 Credits pro Monat). Ohne Key antwortet der Berater nur allgemein und gibt keine Rangliste aus.
- API-Dokumentation: http://localhost:8011/docs
- Datenbank direkt ansehen: PostgreSQL auf `localhost:5441`, Datenbank, Benutzer und Passwort jeweils `bike`.
- Die Ports unterscheiden sich vom AgenticRAG-Abenteuerplaner (5440/8010/5180), damit beide parallel laufen.

> **Warum läuft das Backend im Container?** Auf manchen Windows-Rechnern blockiert die Anwendungssteuerung (Smart App Control) die nativen Postgres-Treiber (`asyncpg`, `psycopg`). Im Linux-Container tritt das nicht auf. Der Ordner `backend/` ist per Volume eingebunden, Codeänderungen werden also sofort neu geladen.

## Tests

```bash
docker compose exec backend sh -c 'ALEMBIC_DATABASE_URL=$TEST_DATABASE_URL alembic upgrade head'   # einmalig
docker compose exec backend pytest -q
```

Die Tests laufen gegen die Datenbank `bike_test`. Sie verwenden ein geskriptetes Modell (`FunctionModel`, siehe `tests/helpers.py`), Fake-Embeddings und eine Relevanzprüfung ohne LLM, rufen also kein echtes LLM auf. Tavily wird mit `respx` gemockt. Geprüft werden unter anderem alle Regeln des Output-Validators (unbekannte URL, Preis nicht auf der Seite, Marke fehlt, Budget überschritten, Motor trotz „ohne Motor“, fehlende Rahmengröße, fehlende Recherche), die Vorab-Phase des Graphen und die SSE-Eventfolge des Chat-Endpunkts.

GitHub Actions führt die Tests in Docker aus (`.github/workflows/docker.yml`), prüft Ruff (`lint.yml`), `.env.example` gegen `config.py` (`config.yml`) und baut das Frontend mit Biome-Prüfung (`frontend.yml`).

## Linting und Formatierung

```bash
docker compose exec backend ruff check --fix .     # Regeln prüfen, Sicheres automatisch beheben
docker compose exec backend ruff format .          # formatieren
cd frontend && npm run lint                        # Biome: Regeln + Formatierung prüfen
cd frontend && npm run format                      # Biome: beheben und formatieren
```

## Wie der Agent arbeitet

**Ablauf als Graph** (`backend/app/agent/graph.py`, LangGraph):

```
START → intake ─┬─(Suchen)────────→ prefetch_web_search ─┐
                ├─(neue Beratung)─→ prefetch_research ───┤  (parallel)
                ├─(sonst)────────────────────────────────┴→ recommender → finish_recommendation
                │                                      | finish_question | finish_answer | finish_error → END
                └─(fachfremd)─────→ reject_off_topic ───────────────────────────────────────────────→ END
```

1. **Intake** (`agent/intake.py`): Ein kleines Modell erkennt Kategorie, Motor ja/nein, Budget, Einsatzzweck und Körpermaße. Es formuliert bis zu 3 erste Websuchen und bei einer neuen Beratung den Auftrag an den Recherche-Agenten. Fachfremde Fragen und Fragen zum Innenleben bekommen eine feste Absage, ohne dass der Berater läuft.
2. **Vorab-Phase:** Websuchen und Recherche laufen parallel. Ihre Ergebnisse stehen als gewöhnliche Tool-Aufrufe im Verlauf des Beraters, der jedes Tool trotzdem selbst aufrufen kann.
3. **Berater** (`agent/recommender.py`): Er sucht weitere Kandidaten mit `search_bikes`, belegt sie mit `read_product_page` (Preis, Ausstattung, Größentabelle), bestimmt die Rahmengrößen mit `recommend_frame_size` und gibt die Rangliste über `final_recommendation` aus. Alternativ stellt er eine Rückfrage (`ask_user`) oder antwortet ohne Rangliste (`answer_user`).

**Ausgabe und Selbstprüfung:**

- **Strukturierte Ausgabe** (`agent/recommendation_schema.py`): `BikeRecommendation` mit 1 bis 5 Rädern. Die Ränge müssen lückenlos sein, und jedes Modell darf nur einmal vorkommen.
- **Output-Validator `check_plausibility`:**
  - Vor der ersten Rangliste muss eine Recherche in der Wissensbasis stattgefunden haben.
  - Jede `price_source_url` muss aus `search_bikes` oder `read_product_page` stammen, auch aus früheren Nachrichten derselben Beratung.
  - Auf dieser Seite müssen Marke und Preis stehen. Die Preiserkennung (`agent/tools/prices.py`) versteht Schreibweisen wie „3.299,00 €“, „EUR 2.499,-“ und „€ 4,999.00“.
  - Der Preis darf höchstens 10 % über dem Budget liegen.
  - Bei „ohne Motor“ darf kein Rad einen Motor haben.
  - Sind Körpermaße bekannt, muss `recommend_frame_size` aufgerufen worden sein, und jedes Rad braucht eine Rahmengröße.
- **Fehler als Ergebnis:** Tools geben Fehler als lesbares Ergebnis zurück (z. B. Tavily-Limit, Seite nicht lesbar), statt Exceptions zu werfen. Der Agent nimmt dann einen anderen Treffer oder antwortet ohne Rangliste.
- **Gesprächsverlauf** je Beratung (`agent_history`), **Streaming** über `POST /api/chat/{id}` als Server-Sent Events (`token`, `tool_call`, `tool_result`, `question`, `answer`, `recommendation`, `error`, `done`).
- **Robuste Ausgabe:** strict-Modus, Tokenlimit und 120 s Zeitlimit je Modellanfrage (`agent/models.py`), Drosselung auf das Anfragelimit (`agent/rate_limit.py`).

## Websuche

`backend/app/agent/tools/web.py`:

- `search_bikes(query)`: Tavily `/search` (Land Deutschland, ohne Video- und Social-Media-Seiten). Liefert je Treffer Titel, URL, Auszug und die darin erkannten Preise.
- `read_product_page(url)`: Tavily `/extract`. Liefert den bereinigten Seitentext (Bilder und Link-Ziele entfernt, höchstens 5.000 Zeichen) und die erkannten Preise.

Antworten bleiben `WEB_CACHE_HOURS` (Standard 24 h) in der Tabelle `web_cache`. Gleiche Anfragen in Folgenachrichten oder Evaluationen kosten so kein Kontingent. Preise und Verfügbarkeit stammen aus dem Web und können sich jederzeit ändern. Die Oberfläche weist darauf hin.

## Rahmengröße

`backend/app/agent/tools/frame_size.py`:

- **Größentabelle:** Ist auf der Produktseite eine Tabelle (Körpergröße je Größe) vorhanden, gibt der Agent sie als `size_chart` mit, und sie hat Vorrang.
- **Faustformeln** je Kategorie, sonst:
  - MTB: Schrittlänge × 0,226 = Zoll
  - Trekking/City: Schrittlänge × 0,66 = cm
  - Rennrad/Gravel: Schrittlänge × 0,665 = cm
- **Schätzung der Schrittlänge:** Fehlt sie, gilt Körpergröße × 0,47.
- **Grenzfälle:** Nahe an einer Größengrenze nennt das Tool beide Größen.
- **Kinderräder:** Laufradgröße nach Körpergröße.
- **Lasten- und Falträder:** Einheitsgröße.

## Wissensbasis

`backend/app/seed/documents/` enthält eigene Kaufratgeber:
- Fahrradkategorien
- Rahmengrößen
- E-Bike-Motoren und Akkus
- Schaltung, Antrieb und Bremsen
- Budget, Kaufzeitpunkt und Kauf

Die Suche übernimmt der **Recherche-Agent** (`agent/research.py`, Tool `research_knowledge`, höchstens 5 Suchen). Jede Suche durchläuft folgende Schritte (`rag/`):
1. Chunking an Markdown-Überschriften, höchstens 1.200 Zeichen je Abschnitt
2. Vektorsuche mit pgvector, Top 4
3. Mindestähnlichkeit 0,5
4. gebündelte Relevanzprüfung durch `RELEVANCE_MODEL`

Erweitern:
- **Als Admin unter *Wissensbasis*** `.md`-, `.txt`- oder `.pdf`-Dateien hochladen.
- **Als Datei:** `.md` nach `seed/documents/` legen und dann `python -m app.seed` ausführen (`--force` liest alle neu ein).

## Qualität messen

**Retrieval** (`evaluation/retrieval.py`, `retrieval_cases.json`): 21 beantwortbare und 4 nicht beantwortbare Fragen zu den Ratgebern. Ausgegeben werden Hit@1/Hit@4, MRR, Recall wie im Agenten, Fehlkontext und ein Schwellen-Sweep.

```bash
docker compose exec backend python -m app.evaluation.retrieval [--grade] [--output /app/report.json]
```

**Berater** (`evaluation/recommender.py`, `recommender_cases.json`): Die Fälle sind Pendler-E-Bike, Gravelbike ohne Motor, Kinderrad, E-Lastenrad und eine unvollständige Anfrage, die eine Rückfrage auslösen soll. Jeder Fall läuft wie im Chat durch den Graphen, mit echten Modellen und echter Websuche.
- **Gemessen:** Laufzeit, Web-Aufrufe, Modellaufrufe, Tokens, Kosten des Beraters und Wiederholungen durch die Selbstprüfung.
- **Geprüft:** Ausgabetyp, 1 bis 5 Räder, belegte Quellen, Kategorie, Budget, Motor und Rahmengröße.

```bash
docker compose exec backend python -m app.evaluation.recommender --repeat 1 --output /app/eval.json
docker compose exec backend python -m app.evaluation.recommender --case gravel-ohne-motor
```

Ein Durchlauf kostet echte Modell- und Tavily-Aufrufe und belegt dasselbe Ratenlimit wie die App.

## Tracing mit Langfuse oder LangSmith

`backend/app/observability.py`, optional. `TRACING_BACKEND` wählt das Ziel (`langfuse`, `langsmith` oder `none`, Standard `langfuse`). Jede Chat-Nachricht landet als Trace `chat-turn`:
- darunter `intake`, der Recherche-Agent `knowledge-researcher` mit `retrieve-knowledge` und `relevance-grader` sowie der Agent `bike-recommender` mit allen Modell- und Tool-Aufrufen
- je Beratung eine Session (Langfuse) bzw. ein Thread (LangSmith) `consultation-<id>`; als Nutzer wird die numerische ID übertragen, nicht die E-Mail
- gescheiterte Tool-Aufrufe sind Warnungen: in Langfuse Level `WARNING`, in LangSmith Metadatum `level=WARNING`

| Backend | Aktiv mit | Export |
|---|---|---|
| `langfuse` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Langfuse-SDK |
| `langsmith` | `LANGSMITH_API_KEY`; Projekt `LANGSMITH_PROJECT`, Region `LANGSMITH_ENDPOINT` (EU: `https://eu.api.smith.langchain.com`) | OTLP an `<LANGSMITH_ENDPOINT>/otel/v1/traces` |

Tests senden nie Traces.

## Konfiguration (`.env`)

Vorlage ist `.env.example`. Sie muss jede Einstellung aus `backend/app/config.py` enthalten (Prüfung: `cd backend && python -m app.check_config`).

| Variable | Bedeutung |
|---|---|
| `LLM_MODEL` | PydanticAI-Modellname des Beraters, z. B. `openai:gpt-5-mini`, `anthropic:claude-sonnet-5-5` |
| `LLM_REQUESTS_PER_MINUTE` | Anfragelimit des API-Keys je Modell; alle Agenten drosseln sich darauf. 0 = nicht drosseln. |
| `LLM_REASONING_EFFORT` | Optional: Reasoning-Aufwand des Beraters (`minimal`, `low`, `medium`, `high`) |
| `INTAKE_MODEL` | Kleines Modell für Kriterien und erste Websuchen (Standard `openai:gpt-4o-mini`). Leer = aus. |
| `RESEARCH_MODEL` | Modell des Recherche-Agenten (Standard `openai:gpt-4.1-mini`) |
| `RELEVANCE_MODEL` | Relevanzprüfung der Wissensbasis-Treffer (Standard `openai:gpt-4o-mini`). Leer = nur Mindestähnlichkeit. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | Bei einem Wechsel: `EMBEDDING_DIM` anpassen, Migration für die Vektorspalte erzeugen, `python -m app.seed --force` |
| `TAVILY_API_KEY` / `TAVILY_API_URL` | Websuche. Ohne Key ist keine Rangliste möglich. |
| `WEB_SEARCH_MAX_RESULTS` | Treffer je Suche (Standard 5, höchstens 10) |
| `WEB_CACHE_HOURS` | Dauer des Zwischenspeichers für Suchergebnisse und Seiten (Standard 24) |
| `HTTP_TIMEOUT_SECONDS` | Zeitlimit für externe Dienste |
| `TRACING_BACKEND` | `langfuse`, `langsmith` oder `none` (siehe Tracing) |
| `LANGFUSE_*` / `LANGSMITH_*` | Optional: Zugangsdaten des Tracing-Backends (siehe oben) |

## Projektstruktur

```
backend/app/
  agent/       graph.py (LangGraph-Ablauf), intake.py (Intake), recommender.py (Berater + Validator),
               research.py (Recherche-Agent), runner.py (Event-Streaming), recommendation_schema.py,
               history.py, deps.py, models.py, rate_limit.py
    tools/     web (Tavily-Suche, Produktseite), prices (Preiserkennung), frame_size, knowledge, research
  rag/         ingest.py, embeddings.py, retrieve.py, relevance.py, attribution.py (Quellen aus dem Verlauf)
  evaluation/  retrieval.py + retrieval_cases.json, recommender.py + recommender_cases.json
  api/         auth, users, consultations, chat (SSE), documents
  models/      user, consultation (+ chat_messages), document (+ chunks), web_cache
  seed/        Admin-Seed, Ratgeber-Dokumente (documents/*.md)
frontend/src/
  pages/       Login/Registrierung, Beratungsliste, Beratung (Chat + Rangliste), Profil (Körpermaße), Admin
  components/  RankingCard, CompareTable, ToolCallBadge, LinkedText, ConfirmButton
```
