import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { SessionSummary } from '../../models/session';

@Component({
  selector: 'app-conversations-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './conversations-page.html',
  styleUrl: './conversations-page.css',
})
export class ConversationsPage implements OnInit {
  sessions: SessionSummary[] = [];
  loading: boolean = false;
  error: string | null = null;
  activeProjectId: string | null = null;

  constructor(
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.activeProjectId = this.userService.getActiveProjectId();
    this.loadSessions();
  }

  loadSessions(): void {
    if (!this.activeProjectId) {
      this.error = 'No active project selected';
      return;
    }

    this.loading = true;
    this.error = null;

    this.agentService.getSessions(this.activeProjectId).subscribe({
      next: (sessions) => {
        this.sessions = sessions;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load sessions';
        console.error(err);
        this.loading = false;
      }
    });
  }

  formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleString();
  }
}
