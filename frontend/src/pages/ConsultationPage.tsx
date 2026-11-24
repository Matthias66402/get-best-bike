import { type FormEvent, type KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, json, streamChat } from "../api/client";
import type { BikeRecommendation, ChatEvent, Consultation, ToolCall } from "../api/types";
import { LinkedText } from "../components/LinkedText";
import { RankingCard } from "../components/RankingCard";
import { ToolCallBadge } from "../components/ToolCallBadge";
import { useSpeechInput } from "../lib/useSpeechInput";

interface LiveMessage {
  content: string;
  tool_calls: ToolCall[];
  error?: string;
}

const EXAMPLES = [
  "Trekking-E-Bike bis 3.500 € zum täglichen Pendeln (15 km, teils Schotter), ich bin 1,75 m groß, Schrittlänge 82 cm",
  "Gravelbike bis 2.000 € für Feierabendrunden und Bikepacking, ohne Motor",
  "Lastenrad für zwei Kinder, Budget 6.000 €, wir wohnen hügelig",
];

export function ConsultationPage() {
  const id = Number(useParams().id);
  const [consultation, setConsultation] = useState<Consultation | null>(null);
  const [recommendation, setRecommendation] = useState<BikeRecommendation | null>(null);
  const [input, setInput] = useState("");
  const [pendingUser, setPendingUser] = useState<string | null>(null);
  const [live, setLive] = useState<LiveMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const chatEnd = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    const data = await api<Consultation>(`/api/consultations/${id}`);
    setConsultation(data);
    setRecommendation(data.recommendation);
  }, [id]);

  useEffect(() => {
    load().catch((e) => setError(e.message));
    return () => abort.current?.abort();
  }, [load]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: Abhängigkeiten sind Auslöser, bei neuen Nachrichten nach unten scrollen
  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [consultation?.messages.length, live, pendingUser]);

  const busy = live !== null;
  // Diktat: Erkannter Text wird an die Eingabe angehängt, gesendet wird erst auf Wunsch des Nutzers
  const speech = useSpeechInput((text) =>
    setInput((value) => `${value.trim() ? `${value.trimEnd()} ` : ""}${text.trim()}`),
  );

  function handleEvent(event: ChatEvent) {
    setLive((current) => {
      const m = current ?? { content: "", tool_calls: [] };
      switch (event.type) {
        case "token":
          return { ...m, content: m.content + event.content };
        case "tool_call":
          return { ...m, tool_calls: [...m.tool_calls, { id: event.id, tool: event.tool, args: event.args }] };
        case "tool_result":
          return {
            ...m,
            tool_calls: m.tool_calls.map((c) =>
              c.id === event.id ? { ...c, result: event.result, failed: event.failed } : c,
            ),
          };
        case "question":
          return { ...m, content: event.question };
        case "answer":
          return { ...m, content: event.content };
        case "error":
          return { ...m, error: event.message };
        default:
          return m;
      }
    });
    if (event.type === "recommendation") setRecommendation(event.recommendation);
  }

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    speech.cancel();
    setInput("");
    setError(null);
    setPendingUser(message);
    setLive({ content: "", tool_calls: [] });
    abort.current = new AbortController();
    try {
      await streamChat(id, message, handleEvent, abort.current.signal);
      await load();
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      setPendingUser(null);
      setLive(null);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    send(input);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  }

  async function saveTitle(title: string) {
    setEditingTitle(false);
    if (!consultation || !title.trim() || title === consultation.title) return;
    setConsultation(await api<Consultation>(`/api/consultations/${id}`, { method: "PATCH", body: json({ title }) }));
  }

  if (error && !consultation) return <div className="page error">{error}</div>;
  if (!consultation) return <div className="page muted">Lade …</div>;

  const messages = consultation.messages;
  const isEmpty = messages.length === 0 && !pendingUser;

  return (
    <div className="planner">
      <section className="chat-pane">
        <div className="chat-header">
          <Link to="/" className="muted small">
            ← Alle Beratungen
          </Link>
          {editingTitle ? (
            <input
              // biome-ignore lint/a11y/noAutofocus: Feld öffnet erst auf Klick zum Umbenennen, Fokus ist erwartet
              autoFocus
              defaultValue={consultation.title}
              onBlur={(e) => saveTitle(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveTitle((e.target as HTMLInputElement).value)}
            />
          ) : (
            <h1>
              <button
                type="button"
                className="title-edit"
                onClick={() => setEditingTitle(true)}
                title="Klicken zum Umbenennen"
              >
                {consultation.title}
              </button>
            </h1>
          )}
        </div>

        <div className="messages">
          {isEmpty && (
            <div className="intro">
              <p>
                Beschreibe, was Du suchst: <strong>Einsatzzweck und Budget</strong>, gern auch Kategorie, mit oder ohne
                Motor und Deine Körpergröße bzw. Schrittlänge für die Rahmengröße. Der Agent sucht live im Web und
                stellt Dir die 1 bis 5 am besten passenden Räder zusammen. Fehlt etwas, fragt er nach.
              </p>
              <div className="examples">
                {EXAMPLES.map((example) => (
                  <button type="button" key={example} className="example" onClick={() => send(example)}>
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="msg user">
                <LinkedText text={m.content} />
              </div>
            ) : (
              <div key={m.id} className="msg assistant">
                {m.tool_calls.length > 0 && (
                  <details className="tool-summary">
                    <summary>{m.tool_calls.length} Tool-Aufrufe</summary>
                    {m.tool_calls.map((c) => (
                      <ToolCallBadge key={c.id} call={c} />
                    ))}
                  </details>
                )}
                <div className="msg-text">
                  <LinkedText text={m.content} />
                </div>
              </div>
            ),
          )}
          {pendingUser && (
            <div className="msg user">
              <LinkedText text={pendingUser} />
            </div>
          )}
          {live && (
            <div className="msg assistant live">
              {live.tool_calls.map((c) => (
                <ToolCallBadge key={c.id} call={c} />
              ))}
              {live.content && (
                <div className="msg-text">
                  <LinkedText text={live.content} />
                </div>
              )}
              {live.error && <p className="error">{live.error}</p>}
              {!live.content && !live.error && (
                <div className="muted small thinking">
                  <span className="spinner" /> Agent arbeitet …
                </div>
              )}
            </div>
          )}
          <div ref={chatEnd} />
        </div>

        {error && <p className="error">{error}</p>}
        {speech.error && <p className="error">{speech.error}</p>}
        <form className="composer" onSubmit={onSubmit}>
          <textarea
            value={speech.interim ? `${input}${input.trim() ? " " : ""}${speech.interim}` : input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              speech.listening
                ? "Ich höre zu …"
                : recommendation
                  ? "Änderungswunsch, z. B. „etwas günstiger“ oder „lieber ohne Motor“ …"
                  : "Beschreibe, was für ein Rad Du suchst …"
            }
            rows={3}
            disabled={busy}
            readOnly={speech.listening}
          />
          {speech.supported && (
            <button
              type="button"
              className={speech.listening ? "mic-button listening" : "mic-button"}
              onClick={speech.listening ? speech.stop : speech.start}
              disabled={busy}
              aria-pressed={speech.listening}
              title={
                speech.listening
                  ? "Aufnahme beenden"
                  : "Spracheingabe (die Erkennung läuft über den Dienst des Browsers, z. B. Google bei Chrome)"
              }
            >
              {speech.listening ? "⏹ Stopp" : "🎤"}
            </button>
          )}
          <button type="submit" className="primary" disabled={busy || !input.trim()}>
            {busy ? "Sucht …" : "Senden"}
          </button>
        </form>
      </section>

      <section className="plan-pane">
        {recommendation ? (
          <RankingCard recommendation={recommendation} />
        ) : (
          <div className="plan-placeholder muted">
            <div className="big-icon" aria-hidden>
              🚲
            </div>
            <p>Hier erscheint Deine Rangliste der 1 bis 5 besten Räder mit Preis, Ausstattung und Rahmengröße.</p>
          </div>
        )}
      </section>
    </div>
  );
}
