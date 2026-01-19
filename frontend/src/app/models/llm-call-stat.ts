export interface LLMCallStat {
  id: string;
  llm_profile_id: string | null;
  name: string | null;
  response_code: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  duration_ms: number | null;
  request_data: any | null;
  response_data: any | null;
  cost: number | null;
  currency: string | null;
  created_at: string;
  updated_at: string;
}
