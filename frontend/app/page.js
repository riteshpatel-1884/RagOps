import ResumePreview from "@/components/ResumePreview";
import AskPanel from "@/components/AskPanel";

export default function Home() {
  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Playground</div>
        <h1>Ask the handbook</h1>
        <p>
          Questions are answered from the Asterion Digital Systems employee handbook (27 pages). Ask anything the
          handbook covers, or something it does not, and check that the answer cites the right page.
        </p>
      </div>
      <div className="split">
        <ResumePreview />
        <AskPanel />
      </div>
    </>
  );
}