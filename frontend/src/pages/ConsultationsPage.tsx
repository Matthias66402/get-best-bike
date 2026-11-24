import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, json } from "../api/client";
import type { Consultation, ConsultationListItem } from "../api/types";
import { ConfirmButton } from "../components/ConfirmButton";
import { formatDateTime } from "../lib/labels";

/** Kleinbuchstaben ohne Akzente, damit "muller" auch "Müller" findet. */
const normalize = (text: string) =>
  text
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();

export function ConsultationsPage() {
  const navigate = useNavigate();
  const [consultations, setConsultations] = useState<ConsultationListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Suchbegriff in der URL: bleibt erhalten, wenn man aus einer Beratung zurückkommt
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get("q") ?? "";
  const filtered = useMemo(() => {
    const words = normalize(query).split(/\s+/).filter(Boolean);
    return (
      consultations?.filter((c) => {
        // Titel und Räder der Rangliste, z. B. "Cube" auch ohne Titelnennung
        const text = normalize([c.title, ...c.bikes].join(" "));
        return words.every((word) => text.includes(word));
      }) ?? null
    );
  }, [consultations, query]);

  function search(text: string) {
    setSearchParams(text ? { q: text } : {}, { replace: true });
  }

  useEffect(() => {
    api<ConsultationListItem[]>("/api/consultations")
      .then(setConsultations)
      .catch((e) => setError(e.message));
  }, []);

  async function createConsultation() {
    try {
      const consultation = await api<Consultation>("/api/consultations", { method: "POST", body: json({}) });
      navigate(`/consultations/${consultation.id}`);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(id: number) {
    await api(`/api/consultations/${id}`, { method: "DELETE" });
    setConsultations((list) => list?.filter((c) => c.id !== id) ?? null);
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Deine Fahrradberatungen</h1>
          <p className="muted">
            Beschreibe in eigenen Worten, was Du suchst – der Agent sucht live und liefert die 1 bis 5 besten Räder.
          </p>
        </div>
        <button type="button" className="primary" onClick={createConsultation}>
          + Neue Beratung
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {consultations === null ? (
        <p className="muted">Lade …</p>
      ) : consultations.length === 0 ? (
        <div className="card empty">
          <p>Noch keine Beratung gestartet.</p>
          <button type="button" className="primary" onClick={createConsultation}>
            Erstes Rad finden
          </button>
        </div>
      ) : (
        <>
          <input
            type="search"
            className="search"
            placeholder="Titel oder Rad suchen, z. B. Gravel oder Cube"
            aria-label="Beratungen durchsuchen"
            value={query}
            onChange={(e) => search(e.target.value)}
          />
          {filtered?.length === 0 && <p className="muted">Keine Beratung passt zu „{query}“.</p>}
          <div className="grid">
            {filtered?.map((c) => (
              <Link key={c.id} to={`/consultations/${c.id}`} className="card adventure-card">
                <div className="row between">
                  <span className={`badge ${c.status}`}>{c.status === "recommended" ? "Empfohlen" : "Entwurf"}</span>
                  <ConfirmButton onConfirm={() => remove(c.id)} />
                </div>
                <h2>{c.title}</h2>
                {c.bikes.length > 0 && (
                  <ol className="small bike-preview">
                    {c.bikes.slice(0, 3).map((bike) => (
                      <li key={bike}>{bike}</li>
                    ))}
                  </ol>
                )}
                <p className="muted small">Zuletzt bearbeitet {formatDateTime(c.updated_at)}</p>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
