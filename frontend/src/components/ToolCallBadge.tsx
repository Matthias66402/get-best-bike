import { useState } from "react";
import type { ToolCall } from "../api/types";
import { CATEGORY_LABELS, hostOf, TOOL_LABELS } from "../lib/labels";

function describeArgs(call: ToolCall): string {
  const a = call.args;
  switch (call.tool) {
    case "search_bikes":
    case "search_knowledge":
      return `„${a.query}“`;
    case "read_product_page":
      return hostOf(String(a.url ?? ""));
    case "recommend_frame_size":
      return [
        CATEGORY_LABELS[a.category as keyof typeof CATEGORY_LABELS] ?? a.category,
        a.model,
        a.height_cm ? `${a.height_cm} cm` : null,
        a.inseam_cm ? `Schritt ${a.inseam_cm} cm` : null,
      ]
        .filter(Boolean)
        .join(", ");
    case "research_knowledge":
      return [CATEGORY_LABELS[a.category as keyof typeof CATEGORY_LABELS] ?? a.category, a.use_case]
        .filter(Boolean)
        .join(", ");
    default:
      return "";
  }
}

export function ToolCallBadge({ call }: { call: ToolCall }) {
  const [open, setOpen] = useState(false);
  const meta = TOOL_LABELS[call.tool] ?? { icon: "🔧", label: call.tool };
  const state = call.result === undefined ? "running" : call.failed ? "failed" : "ok";

  return (
    <div className={`tool-call ${state}`}>
      <button type="button" className="tool-call-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span aria-hidden>{meta.icon}</span>
        <strong>{meta.label}</strong>
        <span className="tool-args">{describeArgs(call)}</span>
        <span className="tool-state" role="img" aria-label={state}>
          {state === "running" ? <span className="spinner" /> : state === "failed" ? "⚠️" : "✓"}
        </span>
      </button>
      {open && (
        <div className="tool-detail">
          <div className="muted small">Argumente</div>
          <pre>{JSON.stringify(call.args, null, 2)}</pre>
          {call.result !== undefined && (
            <>
              <div className="muted small">Ergebnis (gekürzt)</div>
              <pre>{call.result}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}
