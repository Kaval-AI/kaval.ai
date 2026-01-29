
export interface RagResult {
  id: string;
  model: string;
  collection_name: string;
  source_id: string;
  content: string;
  embedding_size: number;
  rag_metadata: any;
  created_at: string;
  updated_at: string;
  similarity: number;
}

export interface RagStats {
  total_entries: number;
  total_collections: number;
  collections: string[];
}
