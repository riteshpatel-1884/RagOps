"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { LayoutDashboard, FlaskConical, BarChart3, FlaskRound } from "lucide-react";
import ThemeToggle from "@/components/ThemeToggle";

// Only real, built pages. Nothing here should ever 404.
const NAV = [
  {
    group: "Overview",
    items: [{ label: "Upload", href: "/workspace", icon: LayoutDashboard }],
  },
  {
    group: "RAG",
    items: [{ label: "Playground", href: "/workspace/playground", icon: FlaskRound }],
  },
  {
    group: "Evaluation",
    items: [
      { label: "Evaluate", href: "/workspace/evaluate", icon: BarChart3 },
      { label: "Experiments", href: "/workspace/experiment", icon: FlaskConical },
    ],
  },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-60 shrink-0 flex-col border-r border-border bg-surface">
      <Link
        href="/"
        className="flex items-center gap-2 border-b border-border px-5 py-4 transition-colors hover:bg-surfaceHover"
      >
        <div className="h-2 w-2 rounded-full bg-accent" />
        <span className="font-mono text-sm tracking-tight text-text">ragops</span>
      </Link>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {NAV.map((section) => (
          <div key={section.group} className="mb-5">
            <div className="mb-1.5 px-2 text-xs text-muted">{section.group}</div>
            <div className="flex flex-col gap-0.5">
              {section.items.map((item) => {
                const Icon = item.icon;
                const active =
                  item.href === "/workspace" ? pathname === "/workspace" : pathname?.startsWith(item.href);

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={clsx(
                      "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors",
                      active ? "bg-accentMuted/30 text-text" : "text-muted hover:bg-surfaceHover hover:text-text"
                    )}
                  >
                    <Icon size={15} />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="flex items-center justify-between border-t border-border px-5 py-3">
        <span className="text-xs text-muted">RAGOps</span>
        <ThemeToggle />
      </div>
    </aside>
  );
}