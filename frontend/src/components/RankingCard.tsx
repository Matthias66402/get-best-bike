import type { BikeRecommendation, RankedBike } from "../api/types";
import { CATEGORY_LABELS, formatPrice, hostOf } from "../lib/labels";
import { CompareTable } from "./CompareTable";

function requirementChips(rec: BikeRecommendation): string[] {
  const r = rec.requirements;
  return [
    r.category ? CATEGORY_LABELS[r.category] : null,
    r.electric === true ? "mit Motor" : r.electric === false ? "ohne Motor" : null,
    r.budget_eur ? `bis ${formatPrice(r.budget_eur)}` : null,
    r.height_cm ? `${r.height_cm} cm` : null,
    r.inseam_cm ? `Schrittlänge ${r.inseam_cm} cm` : null,
    ...r.other_wishes,
  ].filter((chip): chip is string => Boolean(chip));
}

function BikeItem({ bike }: { bike: RankedBike }) {
  const specs = bike.key_specs;
  const facts = [
    specs.motor,
    specs.battery_wh ? `${specs.battery_wh} Wh` : null,
    specs.drivetrain,
    specs.weight_kg ? `${specs.weight_kg.toLocaleString("de-DE")} kg` : null,
  ].filter(Boolean);
  return (
    <li className={`bike card rank-${bike.rank}`}>
      <div className="bike-rank" title={`Platz ${bike.rank}`}>
        {bike.rank}
      </div>
      <div className="bike-body">
        <div className="row between wrap">
          <h3>
            {bike.brand} {bike.model}
            {bike.model_year && <span className="muted small"> ({bike.model_year})</span>}
          </h3>
          <span className="bike-price">{formatPrice(bike.price_eur)}</span>
        </div>
        <div className="row wrap small">
          <span className="chip">{CATEGORY_LABELS[bike.category]}</span>
          {bike.frame_size && <span className="chip">Größe {bike.frame_size}</span>}
        </div>
        <div className="score" title={`Passt zu ${bike.match_score} %`}>
          <div className="score-bar" style={{ width: `${bike.match_score}%` }} />
          <span className="small">{bike.match_score} % passend</span>
        </div>
        <p>{bike.why_fits}</p>
        {facts.length > 0 && <p className="muted small">{facts.join(" · ")}</p>}
        <div className="pros-cons">
          <ul className="pros">
            {bike.pros.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
          <ul className="cons">
            {bike.cons.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
        <a href={bike.price_source_url} target="_blank" rel="noreferrer noopener" className="small text-link">
          Zum Angebot bei {hostOf(bike.price_source_url)} ↗
        </a>
      </div>
    </li>
  );
}

export function RankingCard({ recommendation }: { recommendation: BikeRecommendation }) {
  const chips = requirementChips(recommendation);
  return (
    <div className="plan">
      <section>
        <h2>{recommendation.title}</h2>
        {chips.length > 0 && (
          <div className="chips">
            {chips.map((chip) => (
              <span key={chip} className="chip">
                {chip}
              </span>
            ))}
          </div>
        )}
        <p>{recommendation.summary}</p>
      </section>

      <ol className="ranking">
        {recommendation.ranking.map((bike) => (
          <BikeItem key={bike.rank} bike={bike} />
        ))}
      </ol>

      {recommendation.ranking.length > 1 && (
        <section>
          <h3>Vergleich</h3>
          <CompareTable bikes={recommendation.ranking} />
        </section>
      )}

      {recommendation.buying_tips.length > 0 && (
        <section>
          <h3>Kauftipps</h3>
          <ul>
            {recommendation.buying_tips.map((tip) => (
              <li key={tip}>{tip}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="small">
        <h3>Quellen</h3>
        <ul>
          {recommendation.web_sources.map((s) => (
            <li key={s.url}>
              <a href={s.url} target="_blank" rel="noreferrer noopener" className="text-link">
                {s.title}
              </a>{" "}
              <span className="muted">({hostOf(s.url)})</span>
            </li>
          ))}
          {recommendation.knowledge_sources.map((title) => (
            <li key={title}>Wissensbasis: {title}</li>
          ))}
        </ul>
        <p className="muted">
          Preise und Ausstattung stammen aus einer Live-Websuche und können sich ändern. Bitte vor dem Kauf beim Händler
          prüfen.
        </p>
      </section>
    </div>
  );
}
