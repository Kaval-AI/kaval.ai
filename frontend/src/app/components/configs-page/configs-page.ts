import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { LLMConfig } from '../../models/llm-config';

@Component({
  selector: 'app-configs-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './configs-page.html',
  styleUrl: './configs-page.css',
})
export class ConfigsPage implements OnInit {
  configs: LLMConfig[] = [];
  loading: boolean = false;
  error: string | null = null;
  activeProjectId: string | null = null;

  constructor(
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.activeProjectId = this.userService.getActiveProjectId();
    this.loadConfigs();
  }

  loadConfigs(): void {
    if (!this.activeProjectId) {
      this.error = 'No active project selected';
      return;
    }

    this.loading = true;
    this.error = null;

    this.agentService.getLLMConfigs(this.activeProjectId).subscribe({
      next: (configs) => {
        this.configs = configs;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load LLM configurations';
        console.error(err);
        this.loading = false;
      }
    });
  }

  formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }
}
