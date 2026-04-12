import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MedRAG — AI Healthcare Knowledge Assistant",
  description:
    "Multi-agent RAG system for AI in Healthcare research. Ask questions and get cited, reasoned answers from 60 expert documents.",
  keywords: ["healthcare AI", "RAG", "medical AI", "knowledge base"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
