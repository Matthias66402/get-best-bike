import type { BikeCategory } from "../api/types";

export const CATEGORY_LABELS: Record<BikeCategory, string> = {
  city: "Cityrad",
  trekking: "Trekkingrad",
  road: "Rennrad",
  gravel: "Gravelbike",
  mtb_hardtail: "MTB Hardtail",
  mtb_fully: "MTB Fully",
  cargo: "Lastenrad",
  folding: "Faltrad",
  kids: "Kinderrad",
  other: "Sonstiges",
};

export const TOOL_LABELS: Record<string, { icon: string; label: string }> = {
  search_bikes: { icon: "🔎", label: "Websuche" },
  read_product_page: { icon: "📄", label: "Produktseite" },
  recommend_frame_size: { icon: "📏", label: "Rahmengröße" },
  research_knowledge: { icon: "📚", label: "Recherche" },
  search_knowledge: { icon: "📚", label: "Wissensbasis" },
};

const priceFormat = new Intl.NumberFormat("de-DE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
export const formatPrice = (eur: number | null) => (eur === null ? "Preis unbekannt" : priceFormat.format(eur));

const dateTimeFormat = new Intl.DateTimeFormat("de-DE", { dateStyle: "medium", timeStyle: "short" });
export const formatDateTime = (iso: string) => dateTimeFormat.format(new Date(iso));

/** Hostname ohne "www.", z. B. für Quellenlinks */
export const hostOf = (url: string) => {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
};
