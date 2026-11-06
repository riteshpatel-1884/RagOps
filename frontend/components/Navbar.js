"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Ask Resume" },
  { href: "/evaluation", label: "Evaluation" },
];

export default function Navbar() {
  const path = usePathname();
  return (
    <header className="nav">
      <div className="nav-inner">
        <Link href="/" className="brand">
          <span className="brand-dot" /> RAGOPS
        </Link>
        <nav className="nav-links">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={path === l.href ? "nav-link active" : "nav-link"}
            >
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
