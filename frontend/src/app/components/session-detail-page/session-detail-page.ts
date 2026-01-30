/*
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
