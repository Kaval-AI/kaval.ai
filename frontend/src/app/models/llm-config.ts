export interface LLMConfig {
  id: string;
  name: string;
  provider: string;
  model_name: string;
  base_url: string | null;
  config: any;
  total_cost: number;
  created_at: string;
  updated_at: string;
}
