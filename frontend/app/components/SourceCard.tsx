"use client";

import { SourceDoc } from "../types";

interface Props {
  doc: SourceDoc;
  onClick: () => void;
}

export default function SourceCard({ doc, onClick }: Props) {
  return (
    <div
      className="source-card"
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onClick()}
    >
      <div className="source-card-id">{doc.doc_id}</div>
      <div className="source-card-title">{doc.title}</div>
      <div className="source-card-meta">
        {doc.source} · {doc.date.slice(0, 7)}
      </div>
      <span className={`source-card-type ${doc.source_type}`}>
        {doc.source_type.replace("_", " ")}
      </span>
    </div>
  );
}
