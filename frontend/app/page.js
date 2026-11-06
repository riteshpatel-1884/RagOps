import ResumePreview from "@/components/ResumePreview";
import AskPanel from "@/components/AskPanel";

export default function Home() {
  return (
    <>
      <div className="page-head">
        <h1>Ask my resume</h1>
        <p>
          Read the resume on the left, then ask a question on the right. Pick which RAG version
          (v1, v2, v3) should answer it.
        </p>
      </div>
      <div className="split">
        <ResumePreview />
        <AskPanel />
      </div>
    </>
  );
}
