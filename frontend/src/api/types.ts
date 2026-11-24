export type Role = "user" | "admin";

export interface User {
  id: number;
  email: string;
  display_name: string;
  role: Role;
  is_active: boolean;
  created_at: string;
  /** Körpermaße in cm für die Rahmengröße */
  height_cm: number | null;
  inseam_cm: number | null;
}

export type BikeCategory =
  | "city"
  | "trekking"
  | "road"
  | "gravel"
  | "mtb_hardtail"
  | "mtb_fully"
  | "cargo"
  | "folding"
  | "kids"
  | "other";

export interface Requirements {
  use_case: string;
  category: BikeCategory | null;
  electric: boolean | null;
  budget_eur: number | null;
  height_cm: number | null;
  inseam_cm: number | null;
  other_wishes: string[];
}

export interface KeySpecs {
  weight_kg: number | null;
  drivetrain: string | null;
  motor: string | null;
  battery_wh: number | null;
  suspension: string | null;
  brakes: string | null;
}

export interface RankedBike {
  rank: number;
  brand: string;
  model: string;
  model_year: number | null;
  category: BikeCategory;
  price_eur: number | null;
  price_source_url: string;
  key_specs: KeySpecs;
  frame_size: string | null;
  match_score: number;
  why_fits: string;
  pros: string[];
  cons: string[];
}

export interface WebSource {
  url: string;
  title: string;
}

/** Rangliste der 1-5 besten Räder; Quellen ermittelt das Backend aus dem Agent-Verlauf */
export interface BikeRecommendation {
  title: string;
  summary: string;
  requirements: Requirements;
  ranking: RankedBike[];
  buying_tips: string[];
  sources: string[];
  web_sources: WebSource[];
  knowledge_sources: string[];
}

export interface ToolCall {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  result?: string;
  failed?: boolean;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  tool_calls: ToolCall[];
  created_at: string;
}

export interface ConsultationSummary {
  id: number;
  title: string;
  status: "draft" | "recommended";
  created_at: string;
  updated_at: string;
}

/** Eintrag der Übersichtsliste */
export interface ConsultationListItem extends ConsultationSummary {
  /** Marke und Modell der Rangliste, für Vorschau und Suche */
  bikes: string[];
}

export interface Consultation extends ConsultationSummary {
  recommendation: BikeRecommendation | null;
  messages: ChatMessage[];
}

export interface KnowledgeDocument {
  id: number;
  title: string;
  source: string;
  chunk_count: number;
  created_at: string;
}

export type ChatEvent =
  | { type: "token"; content: string }
  | ({ type: "tool_call" } & ToolCall)
  | { type: "tool_result"; id: string; result: string; failed: boolean }
  | { type: "question"; question: string; missing_fields: string[] }
  | { type: "answer"; content: string }
  | { type: "recommendation"; recommendation: BikeRecommendation }
  | { type: "error"; message: string }
  | { type: "done" };
