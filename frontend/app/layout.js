import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata = {
  title: "RAGOPS",
  description: "RAG experiments: ask documents questions and compare three retrieval setups with measured results.",
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