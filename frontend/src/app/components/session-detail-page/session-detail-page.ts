import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { ChatMessage } from '../../models/chat-message';

@Component({
  selector: 'app-session-detail-page',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './session-detail-page.html',
  styleUrl: './session-detail-page.css',
})
export class SessionDetailPage implements OnInit {
  sessionId: string | null = null;
  projectId: string | null = null;
  messages: ChatMessage[] = [];
  loading: boolean = false;
  error: string | null = null;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.projectId = this.userService.getActiveProjectId();
    this.route.paramMap.subscribe((params) => {
      this.sessionId = params.get('sessionId');
      if (this.sessionId && this.projectId) {
        this.loadMessages();
      } else {
        this.error = 'Invalid session or project';
      }
    });
  }

  loadMessages(): void {
    if (!this.projectId || !this.sessionId) return;

    this.loading = true;
    this.error = null;

    this.agentService.getSessionMessages(this.projectId, this.sessionId).subscribe({
      next: (messages) => {
        this.messages = messages;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load messages';
        console.error(err);
        this.loading = false;
      },
    });
  }

  formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }

  goBack(): void {
    this.router.navigate(['/conversations']);
  }
}
