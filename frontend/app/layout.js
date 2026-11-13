import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata = {
  title: "RAGOPS",
  description: "RAG experiments: ask the Asterion employee handbook questions and see exactly what each retrieval setup used and how it scored.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main className="container">{children}</main>
      </body>
    </html>
  );
}