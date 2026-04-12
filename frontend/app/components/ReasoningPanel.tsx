"use client";

import { useState } from "react";

interface Props {
  trace: string;
}

export default function ReasoningPanel({ trace }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="reasoning-panel">
      <button
        className="reasoning-header"
        onClick={() => setOpen(!open)}
        id="reasoning-toggle"
        aria-expanded={open}
      >
        <span>🧠</span>
        <span>Chain-of-Thought Reasoning</span>
        <span className={`reasoning-chevron ${open ? "open" : ""}`}>▼</span>
      </button>
      {open && (
        <div className="reasoning-body">{trace}</div>
      )}
    </div>
  );
}
