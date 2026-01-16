export interface ChatMessage {
  id: string;
  agent_id: string;
  session_id: string;
  run_id: string | null;
  role: string;
  content: string;
  created_at: string;
  updated_at: string;
}
