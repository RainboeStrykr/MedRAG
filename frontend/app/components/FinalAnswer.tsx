"use client";

import { SourceDoc } from "../types";

interface Props {
  text: string;
  docs: SourceDoc[];
  onDocClick: (doc: SourceDoc) => void;
}

/**
 * Renders the final answer with [DOC-XXX] citations replaced by
 * clickable inline badge chips that open the document modal.
 */
export default function FinalAnswer({ text, docs, onDocClick }: Props) {
  const docMap = new Map<string, SourceDoc>(docs.map((d) => [d.doc_id, d]));

  // Split on citation pattern: [DOC-001], [DOC-052], etc.
  const parts = text.split(/(\[DOC-\d{3}\])/g);

  return (
    <div className="final-answer">
      {parts.map((part, i) => {
        const match = part.match(/^\[DOC-(\d{3})\]$/);
        if (match) {
          const docId = `DOC-${match[1]}`;
          const doc = docMap.get(docId);
          return (
            <span
              key={i}
              className="citation"
              title={doc ? doc.title : docId}
              onClick={(e) => {
                e.stopPropagation();
                if (doc) onDocClick(doc);
              }}
            >
              {docId}
            </span>
          );
        }
        // Regular text — preserve linebreaks
        return (
          <span key={i} style={{ whiteSpace: "pre-wrap" }}>
            {part}
          </span>
        );
      })}
    </div>
  );
}
