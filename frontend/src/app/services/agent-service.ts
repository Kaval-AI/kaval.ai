import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Agent } from '../models/agent';
import { SessionSummary } from '../models/session';
import { ChatMessage } from '../models/chat-message';
import { LLMConfig } from '../models/llm-config';

@Injectable({
  providedIn: 'root'
})
export class AgentService {
  constructor(private http: HttpClient) {}

  getAgentsByProject(projectId: string): Observable<Agent[]> {
    return this.http.get<Agent[]>(`/api/agents/all/${projectId}`);
  }

  getLLMConfigs(projectId: string): Observable<LLMConfig[]> {
    return this.http.get<LLMConfig[]>(`/api/projects/${projectId}/llm-configs`);
  }

  getAgentSvgUrl(projectId: string, agentId: string): string {
    return `/api/agents/svg/${projectId}/${agentId}`;
  }

  getAgentStats(projectId: string, agentId?: string): Observable<any> {
    let url = `/api/agents/stats/${projectId}`;
    if (agentId) {
      url += `?agent_id=${agentId}`;
    }
    return this.http.get<any>(url);
  }

  getSessions(projectId: string, agentId?: string, limit: number = 50, offset: number = 0): Observable<SessionSummary[]> {
    let url = `/api/agents/sessions/${projectId}?limit=${limit}&offset=${offset}`;
    if (agentId) {
      url += `&agent_id=${agentId}`;
    }
    return this.http.get<SessionSummary[]>(url);
  }

  getSessionMessages(projectId: string, sessionId: string): Observable<ChatMessage[]> {
    return this.http.get<ChatMessage[]>(`/api/agents/sessions/${projectId}/${sessionId}/messages`);
  }
}
