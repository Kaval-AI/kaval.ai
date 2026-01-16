import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { SessionSummary } from '../../models/session';
import { Agent } from '../../models/agent';

@Component({
  selector: 'app-conversations-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './conversations-page.html',
  styleUrl: './conversations-page.css',
})
export class ConversationsPage implements OnInit {
  sessions: SessionSummary[] = [];
  agents: Agent[] = [];
  loading: boolean = false;
  error: string | null = null;
  activeProjectId: string | null = null;

  // Pagination & Filtering
  selectedAgentId: string = '';
  limit: number = 20;
  offset: number = 0;
  hasMore: boolean = true;

  constructor(
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.activeProjectId = this.userService.getActiveProjectId();
    this.loadAgents();
    this.loadSessions();
  }

  loadAgents(): void {
    if (!this.activeProjectId) return;
    this.agentService.getAgentsByProject(this.activeProjectId).subscribe({
      next: (agents) => {
        this.agents = agents;
      },
      error: (err) => console.error('Failed to load agents', err)
    });
  }

  loadSessions(reset: boolean = true): void {
    if (!this.activeProjectId) {
      this.error = 'No active project selected';
      return;
    }

    if (reset) {
      this.offset = 0;
      this.sessions = [];
      this.hasMore = true;
    }

    this.loading = true;
    this.error = null;

    const agentId = this.selectedAgentId || undefined;

    this.agentService.getSessions(this.activeProjectId, agentId, this.limit, this.offset).subscribe({
      next: (sessions) => {
        if (reset) {
          this.sessions = sessions;
        } else {
          this.sessions = [...this.sessions, ...sessions];
        }
        this.loading = false;
        if (sessions.length < this.limit) {
          this.hasMore = false;
        }
      },
      error: (err) => {
        this.error = 'Failed to load sessions';
        console.error(err);
        this.loading = false;
      }
    });
  }

  nextPage(): void {
    if (this.loading || !this.hasMore) return;
    this.offset += this.limit;
    this.loadSessions(false);
  }

  onAgentChange(): void {
    this.loadSessions(true);
  }

  formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${day}-${month}-${year} ${hours}:${minutes}`;
  }
}
