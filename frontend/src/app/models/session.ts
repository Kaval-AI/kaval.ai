export interface SessionSummary {
  session_id: string;
  agent_id: string;
  agent_name: string;
  runs_count: number;
  tasks_count: number;
  messages_count: number;
  first_message: string | null;
  last_message: string | null;
  created_at: string;
  updated_at: string;
}
