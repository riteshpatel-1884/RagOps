import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata = {
  title: "RAGOPS",
  description: "RAG evaluation platform: compare experiments and ask questions from your resume.",
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
