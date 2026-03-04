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
import { ActivatedRoute } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { LLMCallStat } from '../../models/llm-call-stat';
import { JsonTreeComponent } from '../json-tree/json-tree';

@Component({
  selector: 'app-llm-call-stats-page',
  standalone: true,
  imports: [CommonModule, JsonTreeComponent],
  templateUrl: './llm-call-stats-page.html',
  styleUrl: './llm-call-stats-page.css'
})
export class LlmCallStatsPage implements OnInit {
  callType: string | null = null;
  projectId: string | null = null;
  stats: LLMCallStat[] = [];
  loading = false;
  error: string | null = null;
  limit = 20;
  offset = 0;
  hasMore = true;

  constructor(
    private route: ActivatedRoute,
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      this.callType = params['call_type'] || null;
      this.userService.userDetails.subscribe(user => {
        if (user && user.active_project_id) {
          this.projectId = user.active_project_id;
          this.offset = 0;
          this.stats = [];
          this.loadStats();
        }
      });
    });
  }

  loadStats(): void {
    if (!this.projectId) return;

    this.loading = true;
    this.agentService.getLLMCallStats(this.projectId, this.callType || undefined, this.limit, this.offset)
      .subscribe({
        next: (data) => {
          this.stats = [...this.stats, ...data];
          this.loading = false;
          this.hasMore = data.length === this.limit;
        },
        error: (err) => {
          this.error = 'Failed to load LLM call stats';
          this.loading = false;
          console.error(err);
        }
      });
  }

  loadMore(): void {
    if (this.loading || !this.hasMore) return;
    this.offset += this.limit;
    this.loadStats();
  }

  formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleString();
  }
}
