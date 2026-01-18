export interface LLMConfig {
  id: string;
  name: string;
  provider: string;
  model_name: string;
  base_url: string | null;
  default_mode: string | null;
  total_cost: number;
  created_at: string;
  updated_at: string;
}
