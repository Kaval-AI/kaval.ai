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
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
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
  totalSessions: number = 0;

  // Pagination & Filtering
  selectedAgentId: string = '';
  searchText: string = '';
  startDate: string = '';
  endDate: string = '';
  limit: number = 20;
  offset: number = 0;
  hasMore: boolean = true;

  constructor(
    private agentService: AgentService,
    private userService: UserService,
    private route: ActivatedRoute,
    private router: Router
  ) {
    // Default to last 7 days
    const now = new Date();
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(now.getDate() - 7);

    // Set to start of day for 7 days ago, end of day for today
    sevenDaysAgo.setHours(0, 0, 0, 0);
    now.setHours(23, 59, 59, 999);

    this.startDate = this.toLocalISOString(sevenDaysAgo);
    this.endDate = this.toLocalISOString(now);
  }

  private toLocalISOString(date: Date): string {
    const tzoffset = date.getTimezoneOffset() * 60000;
    return new Date(date.getTime() - tzoffset).toISOString().slice(0, 16);
  }

  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      if (params['agentId']) {
        this.selectedAgentId = params['agentId'];
      }
    });

    this.userService.userDetails.subscribe(user => {
      if (user && user.active_project_id) {
        const newProjectId = user.active_project_id !== 'None' ? user.active_project_id : null;
        if (newProjectId !== this.activeProjectId) {
          this.activeProjectId = newProjectId;
          this.loadAgents();
          this.loadSessions();
        }
      }
    });
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
    const search = this.searchText || undefined;
    const startDate = this.startDate ? new Date(this.startDate).toISOString() : undefined;
    const endDate = this.endDate ? new Date(this.endDate).toISOString() : undefined;

    this.agentService.getSessions(this.activeProjectId, agentId, search, startDate, endDate, this.limit, this.offset).subscribe({
      next: (data) => {
        const sessions = data.sessions;
        this.totalSessions = data.total_count;
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

  onFilterChange(): void {
    this.loadSessions(true);
  }

  viewSession(sessionId: string): void {
    this.router.navigate(['/conversations', sessionId]);
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
