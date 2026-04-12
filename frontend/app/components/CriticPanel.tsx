"use client";

import { useState } from "react";

interface Props {
  critique: string;
}

export default function CriticPanel({ critique }: Props) {
  const [open, setOpen] = useState(false);

  const hasIssues = !critique.toLowerCase().includes("no issues found");

  return (
    <div className="critic-panel">
      <button
        className="critic-header"
        onClick={() => setOpen(!open)}
        id="critic-toggle"
        aria-expanded={open}
      >
        <span>{hasIssues ? "⚠️" : "✅"}</span>
        <span>
          Critic Agent — {hasIssues ? "Issues flagged" : "All citations verified"}
        </span>
        <span
          style={{
            marginLeft: "auto",
            fontSize: "0.7rem",
            transition: "transform 250ms ease",
            transform: open ? "rotate(180deg)" : "none",
          }}
        >
          ▼
        </span>
      </button>
      {open && (
        <div className="critic-body">{critique}</div>
      )}
    </div>
  );
}
