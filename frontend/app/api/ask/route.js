import { NextResponse } from "next/server";

// Change these to match your Python backend.
const BASE = process.env.RAG_API_URL || "http://localhost:8000";
const ENDPOINT = process.env.RAG_API_PATH || "/ask";

export async function POST(req) {
  try {
    const { question, version } = await req.json();
    if (!question) return NextResponse.json({ error: "Question is required" }, { status: 400 });

    const res = await fetch(`${BASE}${ENDPOINT}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, version }),
      cache: "no-store",
    });

    if (!res.ok) {
      const t = await res.text();
      return NextResponse.json({ error: `Backend error ${res.status}: ${t}` }, { status: 502 });
    }

    const data = await res.json();
    return NextResponse.json({
      answer: data.answer ?? data.response ?? data.result ?? "No answer returned.",
      sources: data.sources ?? data.contexts ?? data.chunks ?? [],
    });
  } catch (e) {
    return NextResponse.json(
      { error: `Could not reach the RAG backend (${BASE}). Is it running?` },
      { status: 500 }
    );
  }
}
