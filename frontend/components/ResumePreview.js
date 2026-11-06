"use client";
import { useEffect, useState } from "react";

export default function ResumePreview() {
  const [files, setFiles] = useState([]);
  const [active, setActive] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/resumes")
      .then((r) => r.json())
      .then((d) => {
        setFiles(d.files || []);
        if (d.files?.length) setActive(d.files[0]);
      })
      .catch(() => setFiles([]))
      .finally(() => setLoading(false));
  }, []);

  const src = active ? `/resume/${encodeURIComponent(active)}` : "";

  return (
    <section className="card preview">
      <div className="card-head">
        <h2>Resume preview</h2>
        {src && (
          <a className="btn-ghost" href={src} target="_blank" rel="noreferrer">
            Open in new tab ↗
          </a>
        )}
      </div>

      {files.length > 1 && (
        <select className="select" value={active} onChange={(e) => setActive(e.target.value)}>
          {files.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
      )}

      {loading ? (
        <div className="empty">Loading resume…</div>
      ) : src ? (
        <iframe className="pdf" src={`${src}#view=FitH`} title="Resume preview" />
      ) : (
        <div className="empty">
          No PDF found. Put your resume in <code>public/resume/</code> (e.g. <code>resume.pdf</code>).
        </div>
      )}
    </section>
  );
}
