import type { RankedBike } from "../api/types";
import { formatPrice } from "../lib/labels";

const ROWS: { label: string; value: (bike: RankedBike) => string | null }[] = [
  { label: "Preis", value: (b) => formatPrice(b.price_eur) },
  { label: "Passend", value: (b) => `${b.match_score} %` },
  { label: "Rahmengröße", value: (b) => b.frame_size },
  {
    label: "Gewicht",
    value: (b) => (b.key_specs.weight_kg ? `${b.key_specs.weight_kg.toLocaleString("de-DE")} kg` : null),
  },
  { label: "Antrieb", value: (b) => b.key_specs.drivetrain },
  { label: "Motor", value: (b) => b.key_specs.motor },
  { label: "Akku", value: (b) => (b.key_specs.battery_wh ? `${b.key_specs.battery_wh} Wh` : null) },
  { label: "Federung", value: (b) => b.key_specs.suspension },
  { label: "Bremsen", value: (b) => b.key_specs.brakes },
];

/** Daten der Räder nebeneinander; Zeilen ohne einen einzigen Wert entfallen. */
export function CompareTable({ bikes }: { bikes: RankedBike[] }) {
  const rows = ROWS.filter((row) => bikes.some((bike) => row.value(bike)));
  return (
    <div className="card table-wrap">
      <table className="compare">
        <thead>
          <tr>
            <th />
            {bikes.map((bike) => (
              <th key={bike.rank}>
                {bike.rank}. {bike.brand} {bike.model}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">{row.label}</th>
              {bikes.map((bike) => (
                <td key={bike.rank}>{row.value(bike) ?? "–"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
