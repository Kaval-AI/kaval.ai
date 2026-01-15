import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Agent } from '../../models/agent';

@Component({
  selector: 'app-agents-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './agents-page.html',
  styleUrl: './agents-page.css'
})
export class AgentsPage implements OnInit {
  agents: Agent[] = [];
  loading: boolean = false;
  error: string | null = null;
  selectedAgent: Agent | null = null;
  activeProjectId: string | null = null;

  constructor(
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.activeProjectId = this.userService.getActiveProjectId();
    this.loadAgents();
  }

  private loadAgents(): void {
    if (!this.activeProjectId) {
      this.error = 'No active project selected';
      return;
    }

    this.loading = true;
    this.error = null;
    console.log(`Loading agents for project ${this.activeProjectId}`);

    this.agentService.getAgentsByProject(this.activeProjectId).subscribe({
      next: (agents) => {
        this.agents = agents;
        this.loading = false;
        if (this.agents.length > 0) {
          this.selectedAgent = this.agents[0];
        }
      },
      error: (err) => {
        this.error = 'Failed to load agents';
        console.error(err);
        this.loading = false;
      }
    });
  }

  selectAgent(agent: Agent): void {
    this.selectedAgent = agent;
  }

  getSvgUrl(agentId: string): string {
    if (!this.activeProjectId) return '';
    return this.agentService.getAgentSvgUrl(this.activeProjectId, agentId);
  }
}
