import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Agent } from '../models/agent';
import { SessionSummary } from '../models/session';
import { ChatMessage } from '../models/chat-message';
import { LLMCallStat } from '../models/llm-call-stat';

@Injectable({
  providedIn: 'root'
})
export class AgentService {
  constructor(private http: HttpClient) {}

  getAgentsByProject(projectId: string): Observable<Agent[]> {
    return this.http.get<Agent[]>(`/api/agents/all/${projectId}`);
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

  getSummaryStats(projectId: string, agentId?: string): Observable<any> {
    let url = `/api/agents/summary-stats/${projectId}`;
    if (agentId) {
      url += `?agent_id=${agentId}`;
    }
    return this.http.get<any>(url);
  }

  getSessions(projectId: string, agentId?: string, limit: number = 50, offset: number = 0): Observable<any> {
    let url = `/api/agents/sessions/${projectId}?limit=${limit}&offset=${offset}`;
    if (agentId) {
      url += `&agent_id=${agentId}`;
    }
    return this.http.get<any>(url);
  }

  getSessionMessages(projectId: string, sessionId: string): Observable<ChatMessage[]> {
    return this.http.get<ChatMessage[]>(`/api/agents/sessions/${projectId}/${sessionId}/messages`);
  }

  getLLMCallStats(projectId: string, callType?: string, limit: number = 50, offset: number = 0): Observable<LLMCallStat[]> {
    let url = `/api/projects/${projectId}/llm-call-stats?limit=${limit}&offset=${offset}`;
    if (callType) {
      url += `&call_type=${callType}`;
    }
    return this.http.get<LLMCallStat[]>(url);
  }
}
