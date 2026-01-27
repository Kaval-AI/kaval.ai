export interface EmbeddingConfig {
  id: string;
  name: string;
  provider: string;
  model_name: string;
  base_url: string | null;
  embedding_size: number | null;
  config: any;
  created_at: string;
  updated_at: string;
}

export interface RagResult {
  id: string;
  embedding_profile_id: string;
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
