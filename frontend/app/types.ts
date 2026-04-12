// types.ts — Shared TypeScript interfaces

export interface SourceDoc {
  doc_id: string;
  title: string;
  source: string;
  source_type: string;
  url: string;
  date: string;
  tags: string[];
}

export interface ChatApiResponse {
  query_type: string;          // "simple" | "complex"
  sub_queries: string[];
  retrieved_docs: SourceDoc[];
  reasoning_trace: string;
  final_answer: string;
  critique: string | null;
  verified_answer: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  // Only present for assistant messages
  response?: ChatApiResponse;
  isLoading?: boolean;
}
