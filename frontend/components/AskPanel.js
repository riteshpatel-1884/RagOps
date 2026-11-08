"use client";
import { useEffect, useRef, useState } from "react";

const VERSION_INFO = {
  v1: { title: "Baseline", detail: "raw PDF blocks, top 3 chunks" },
  v2: { title: "Smaller retriever", detail: "raw PDF blocks, top 1 chunk" },
  v3: { title: "Merged chunks + cleaner search", detail: "~700-character chunks, top 3 chunks" },
};

const SUGGESTIONS = [
  {
    group: "Resume",
    items: ["What are the candidate's key skills?", "Summarise the candidate's education.", "What projects are listed in the resume?"],
  },
  {
    group: "Company handbook",
    items: ["What are the standard office hours at Nexora for full-time employees?", "How many days of casual leave are allowed?"],
  },
  {
    group: "Not in the documents (should say so)",
    items: ["What is the candidate's current salary?"],
  },
];

export default function AskPanel() {
  const [version, setVersion] = useState("v3");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  async function ask(q) {
    const text = (q ?? question).trim();
    if (!text || busy) return;
    setQuestion("");
    setMessages((m) => [...m, { role: "user", text }]);
    setBusy(true);
    const started = performance.now();
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, version }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setMessages((m) => [
        ...m,
        { role: "bot", text: data.answer, sources: data.sources, version, ms: Math.round(performance.now() - started) },
      ]);
    } catch (err) {
      setMessages((m) => [...m, { role: "bot", error: true, text: err.message, version }]);
    } finally {
      setBusy(false);
    }
  }

  const info = VERSION_INFO[version];

  return (
    <section className="card chat">
      <div className="chat-top">
        <div className="card-head">
          <h2>Ask a question</h2>
          <div className="seg" role="tablist" aria-label="RAG version">
            {Object.keys(VERSION_INFO).map((v) => (
              <button key={v} className={version === v ? "seg-btn active" : "seg-btn"} onClick={() => setVersion(v)}>
                {v}
              </button>
            ))}
          </div>
        </div>
        <p className="version-note">
          <b>{version} · {info.title}</b> — {info.detail}
        </p>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div>
            <div className="suggest-title">Try a question</div>
            {SUGGESTIONS.map((g) => (
              <div key={g.group}>
                <div className="suggest-group">{g.group}</div>
                <div className="chips">
                  {g.items.map((s) => (
                    <button key={s} className="chip" onClick={() => ask(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role} ${m.error ? "error" : ""}`}>
            <div className="bubble">{m.text}</div>
            {m.role === "bot" && !m.error && (
              <div className="meta">
                <span className="badge">{m.version}</span>
                <span>{(m.ms / 1000).toFixed(1)} s</span>
              </div>
            )}
            {m.sources?.length > 0 && (
              <details className="sources">
                <summary>{m.sources.length} retrieved chunk{m.sources.length > 1 ? "s" : ""}</summary>
                {m.sources.map((s, j) => (
                  <pre key={j}>{typeof s === "string" ? s : s.text || JSON.stringify(s, null, 2)}</pre>
                ))}
              </details>
            )}
          </div>
        ))}

        {busy && (
          <div className="msg bot">
            <div className="bubble typing">
              <span className="dots"><span /><span /><span /></span>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="composer">
        <input
          className="input"
          placeholder="Ask about the resume or the handbook…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
        />
        <button className="btn" onClick={() => ask()} disabled={busy || !question.trim()}>
          Ask
        </button>
      </div>
    </section>
  );
}