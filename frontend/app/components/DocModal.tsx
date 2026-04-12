"use client";

import { SourceDoc } from "../types";

interface Props {
  doc: SourceDoc;
  onClose: () => void;
}

export default function DocModal({ doc, onClose }: Props) {
  // Close on overlay click
  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onClose();
  };

  return (
    <div
      className="modal-overlay"
      onClick={handleOverlayClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      <div className="modal">
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "10px",
            marginBottom: "12px",
          }}
        >
          <span
            className="font-mono text-accent"
            style={{ fontSize: "0.78rem", paddingTop: "3px", flexShrink: 0 }}
          >
            {doc.doc_id}
          </span>
          <h2 id="modal-title" className="modal-title">
            {doc.title}
          </h2>
        </div>

        <div className="modal-meta">
          <span className="modal-meta-item">📅 {doc.date.slice(0, 10)}</span>
          <span className="modal-meta-item">·</span>
          <span className="modal-meta-item">📰 {doc.source}</span>
          <span className="modal-meta-item">·</span>
          <span
            className={`source-card-type ${doc.source_type}`}
            style={{ fontSize: "0.72rem", padding: "2px 8px" }}
          >
            {doc.source_type.replace("_", " ")}
          </span>
        </div>

        {doc.url && (
          <div className="modal-url">
            🔗{" "}
            <a href={doc.url} target="_blank" rel="noopener noreferrer">
              {doc.url}
            </a>
          </div>
        )}

        {doc.tags.length > 0 && (
          <div className="modal-tags">
            {doc.tags.map((tag) => (
              <span key={tag} className="modal-tag">
                #{tag}
              </span>
            ))}
          </div>
        )}

        <button
          id="modal-close-btn"
          className="modal-close"
          onClick={onClose}
        >
          Close
        </button>
      </div>
    </div>
  );
}
