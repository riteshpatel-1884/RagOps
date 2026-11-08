import ResumePreview from "@/components/ResumePreview";
import AskPanel from "@/components/AskPanel";

export default function Home() {
  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Playground</div>
        <h1>Ask the documents</h1>
        <p>
          Questions are answered from a resume and a company HR handbook. Choose a RAG version
          (v1, v2 or v3) and compare how each one answers the same question.
        </p>
      </div>
      <div className="split">
        <ResumePreview />
        <AskPanel />
      </div>
    </>
  );
}