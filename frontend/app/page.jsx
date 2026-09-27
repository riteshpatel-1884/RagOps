import Link from "next/link";
import {
  ArrowRight,
  LayoutDashboard,
  FlaskRound,
  BarChart3,
  FlaskConical,
} from "lucide-react";
import ThemeToggle from "@/components/ThemeToggle";

const PHASES = [
  {
    icon: LayoutDashboard,
    title: "Upload & label",
    description:
      "Bring your own corpus and build a labeled test dataset to score every future pipeline change against.",
  },
  {
    icon: FlaskRound,
    title: "Playground",
    description: "Sanity-check a config with one-off questions before committing to a full evaluation run.",
  },
  {
    icon: BarChart3,
    title: "Evaluate",
    description: "Score one pipeline config on retrieval metrics (Recall, Precision, MRR, NDCG) and generation quality.",
  },
  {
    icon: FlaskConical,
    title: "Experiment",
    description: "Sweep chunk sizes, overlaps, embedders and generators across a full grid, ranked automatically.",
  },
];

export default function HomePage() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-bg">
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-[480px]"
        style={{
          background:
            "radial-gradient(60% 60% at 50% 0%, rgb(var(--color-accent) / 0.16) 0%, rgb(var(--color-accent) / 0) 70%)",
        }}
      />

      <div className="absolute right-6 top-6 z-10">
        <ThemeToggle />
      </div>

      <div className="relative mx-auto flex max-w-5xl flex-col items-center px-6 pb-20 pt-28 text-center">
        <div className="mb-6 flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs text-muted">
          <div className="h-1.5 w-1.5 rounded-full bg-accent" />
          Production RAG evaluation
        </div>

        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-text sm:text-5xl">
          Know exactly why your RAG pipeline underperforms — and what to change.
        </h1>

        <p className="mt-5 max-w-2xl text-base leading-relaxed text-muted sm:text-lg">
          RAGOps evaluates retrieval and generation quality and sweeps pipeline configs at scale so you can find
          the best-performing setup for your data.
        </p>

        <div className="mt-9 flex flex-col items-center gap-3 sm:flex-row">
          <Link
            href="/workspace"
            className="group flex items-center gap-2 rounded-md bg-brand-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-brand-700"
          >
            Enter workspace
            <ArrowRight size={15} className="transition-transform group-hover:translate-x-0.5" />
          </Link>
          <a
            href="#phases"
            className="rounded-md border border-border bg-surface px-5 py-2.5 text-sm font-medium text-text transition-colors hover:bg-surfaceHover"
          >
            See how it works
          </a>
        </div>
      </div>

      <div id="phases" className="relative mx-auto max-w-5xl px-6 pb-24">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PHASES.map((phase) => {
            const Icon = phase.icon;
            return (
              <div
                key={phase.title}
                className="rounded-xl border border-border bg-surface p-5 shadow-card transition-colors hover:bg-surfaceHover"
              >
                <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-md bg-accentMuted/30 text-accent">
                  <Icon size={17} />
                </div>
                <h3 className="text-sm font-semibold text-text">{phase.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-muted">{phase.description}</p>
              </div>
            );
          })}
        </div>
      </div>

      <div className="relative border-t border-border py-6 text-center text-xs text-muted">
        Built on LangChain · Qdrant · offline-first by default
      </div>
    </div>
  );
}