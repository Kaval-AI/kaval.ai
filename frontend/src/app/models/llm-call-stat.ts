export interface LLMCallStat {
  id: string;
  call_type: string;
  model: string;
  agent_id: string | null;
  response_code: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  duration_seconds: number | null;
  request_data: any | null;
  response_data: any | null;
  cost: number | null;
  currency: string | null;
  created_at: string;
  updated_at: string;
}
