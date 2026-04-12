"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Message, SourceDoc, ChatApiResponse } from "./types";
import ReasoningPanel from "./components/ReasoningPanel";
import CriticPanel from "./components/CriticPanel";
import SourceCard from "./components/SourceCard";
import DocModal from "./components/DocModal";
import FinalAnswer from "./components/FinalAnswer";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const SUGGESTIONS = [
  "What AI approaches are most impactful for patient outcomes?",
  "How does federated learning protect patient privacy in healthcare AI?",
  "Compare Google, Microsoft, and Amazon's healthcare AI strategies.",
  "What ethical frameworks exist for responsible healthcare AI deployment?",
  "What safety concerns exist with LLMs in clinical practice?",
  "How is AI addressing health equity and algorithmic bias?",
];

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [useCritic, setUseCritic] = useState(true);
  const [activeSources, setActiveSources] = useState<SourceDoc[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<SourceDoc | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
    }
  };

  const buildHistory = useCallback(() => {
    return messages
      .filter((m) => !m.isLoading)
      .slice(-8) // last 4 exchanges
      .map((m) => ({
        role: m.role,
        content: m.role === "assistant"
          ? (m.response?.verified_answer || m.content)
          : m.content,
      }));
  }, [messages]);

  const sendMessage = useCallback(
    async (query: string) => {
      if (!query.trim() || loading) return;

      const userMsg: Message = {
        id: Date.now().toString(),
        role: "user",
        content: query.trim(),
      };
      const loadingMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        isLoading: true,
      };

      setMessages((prev) => [...prev, userMsg, loadingMsg]);
      setInput("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
      setLoading(true);

      try {
        const history = buildHistory();
        const res = await fetch(`${API_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query.trim(),
            conversation_history: history,
            use_critic: useCritic,
          }),
        });

        if (!res.ok) {
          const err = await res.text();
          throw new Error(`API error ${res.status}: ${err}`);
        }

        const data: ChatApiResponse = await res.json();
        const displayAnswer = data.verified_answer || data.final_answer;

        const assistantMsg: Message = {
          id: loadingMsg.id,
          role: "assistant",
          content: displayAnswer,
          response: data,
        };

        setMessages((prev) =>
          prev.map((m) => (m.id === loadingMsg.id ? assistantMsg : m))
        );
        setActiveSources(data.retrieved_docs);
      } catch (err: any) {
        const errorMsg: Message = {
          id: loadingMsg.id,
          role: "assistant",
          content: `⚠️ Error: ${err.message}. Make sure the MedRAG backend is running on port 8000.`,
        };
        setMessages((prev) =>
          prev.map((m) => (m.id === loadingMsg.id ? errorMsg : m))
        );
      } finally {
        setLoading(false);
      }
    },
    [loading, useCritic, buildHistory]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const clearConversation = () => {
    setMessages([]);
    setActiveSources([]);
  };

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-logo">
          <div className="logo-icon">🧬</div>
          <div>
            <div className="header-title">MedRAG</div>
            <div className="header-subtitle">Healthcare AI Knowledge System</div>
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: "12px", alignItems: "center" }}>
          {messages.length > 0 && (
            <button className="clear-btn" onClick={clearConversation}>
              Clear
            </button>
          )}
          <div className="header-badge">
            <span className="status-dot" />
            Qwen3:8b · Ollama
          </div>
        </div>
      </header>

      {/* ── Main layout ── */}
      <div className="main-layout">
        {/* ── Chat column ── */}
        <div className="chat-column">
          <div className="messages-area">
            <div className="messages-inner">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">🩺</div>
                  <h1 className="empty-title">Ask about AI in Healthcare</h1>
                  <p className="empty-desc">
                    A multi-agent RAG system over 60 expert documents. Every answer
                    includes cited sources, reasoning traces, and critic verification.
                  </p>
                  <div className="suggestion-grid">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s}
                        className="suggestion-card"
                        onClick={() => sendMessage(s)}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((msg) => (
                  <MessageRow
                    key={msg.id}
                    msg={msg}
                    onDocClick={setSelectedDoc}
                    onSourcesUpdate={setActiveSources}
                  />
                ))
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* ── Input area ── */}
          <div className="input-area">
            <div className="input-inner">
              <div className="input-controls">
                <textarea
                  ref={textareaRef}
                  id="chat-input"
                  className="input-box"
                  placeholder="Ask a question about AI in healthcare… (Shift+Enter for new line)"
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  disabled={loading}
                  rows={1}
                />
                <button
                  id="send-btn"
                  className="send-btn"
                  onClick={() => sendMessage(input)}
                  disabled={loading || !input.trim()}
                  title="Send message"
                >
                  {loading ? "⏳" : "↑"}
                </button>
              </div>
              <div className="input-options">
                <label className="toggle-label" htmlFor="critic-toggle">
                  <input
                    id="critic-toggle"
                    type="checkbox"
                    checked={useCritic}
                    onChange={(e) => setUseCritic(e.target.checked)}
                  />
                  Critic Agent
                </label>
                <span className="input-hint">60 docs · hybrid retrieval · 3-agent pipeline</span>
              </div>
            </div>
          </div>
        </div>

        {/* ── Sources sidebar ── */}
        <aside className="sources-panel">
          <div className="sources-header">Retrieved Sources</div>
          {activeSources.length === 0 ? (
            <div className="sources-empty">
              Sources will appear here after your first query.
            </div>
          ) : (
            activeSources.map((doc) => (
              <SourceCard
                key={doc.doc_id}
                doc={doc}
                onClick={() => setSelectedDoc(doc)}
              />
            ))
          )}
        </aside>
      </div>

      {/* ── Doc modal ── */}
      {selectedDoc && (
        <DocModal doc={selectedDoc} onClose={() => setSelectedDoc(null)} />
      )}
    </div>
  );
}

/* ── MessageRow sub-component ── */
function MessageRow({
  msg,
  onDocClick,
  onSourcesUpdate,
}: {
  msg: Message;
  onDocClick: (doc: SourceDoc) => void;
  onSourcesUpdate: (docs: SourceDoc[]) => void;
}) {
  const isUser = msg.role === "user";

  const handleClick = () => {
    if (msg.response?.retrieved_docs) {
      onSourcesUpdate(msg.response.retrieved_docs);
    }
  };

  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`} onClick={handleClick}>
      <div className={`avatar ${isUser ? "user" : "assistant"}`}>
        {isUser ? "👤" : "🤖"}
      </div>
      <div className="message-content">
        {/* Loading state */}
        {msg.isLoading ? (
          <div className="loading-bubble">
            <div className="loading-dots">
              <div className="loading-dot" />
              <div className="loading-dot" />
              <div className="loading-dot" />
            </div>
            <span className="loading-text">Agents reasoning…</span>
          </div>
        ) : isUser ? (
          <div className="message-bubble">{msg.content}</div>
        ) : (
          <>
            {/* Query type + sub-queries */}
            {msg.response && (
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
                  <span className={`query-type-badge ${msg.response.query_type}`}>
                    {msg.response.query_type === "simple" ? "⚡ Simple" : "🔗 Complex"}
                  </span>
                  <span style={{ fontSize: "0.72rem", color: "var(--clr-text-muted)" }}>
                    {msg.response.retrieved_docs.length} docs retrieved
                  </span>
                </div>
                {msg.response.query_type === "complex" && msg.response.sub_queries.length > 1 && (
                  <div className="sub-queries">
                    {msg.response.sub_queries.map((sq, i) => (
                      <span key={i} className="sub-query-chip">
                        {i + 1}. {sq}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Reasoning trace */}
            {msg.response?.reasoning_trace && (
              <ReasoningPanel trace={msg.response.reasoning_trace} />
            )}

            {/* Final answer */}
            <div className="message-bubble">
              <FinalAnswer
                text={msg.response?.verified_answer || msg.content}
                docs={msg.response?.retrieved_docs || []}
                onDocClick={onDocClick}
              />
            </div>

            {/* Critic */}
            {msg.response?.critique && (
              <CriticPanel critique={msg.response.critique} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
