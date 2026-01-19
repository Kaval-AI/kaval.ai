import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { LLMCallStat } from '../../models/llm-call-stat';
import { JsonTreeComponent } from '../json-tree/json-tree';

@Component({
  selector: 'app-llm-call-stats-page',
  standalone: true,
  imports: [CommonModule, RouterLink, JsonTreeComponent],
  templateUrl: './llm-call-stats-page.html',
  styleUrl: './llm-call-stats-page.css'
})
export class LlmCallStatsPage implements OnInit {
  llmProfileId: string | null = null;
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
      this.llmProfileId = params['llm_profile_id'] || null;
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
    this.agentService.getLLMCallStats(this.projectId, this.llmProfileId || undefined, this.limit, this.offset)
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
