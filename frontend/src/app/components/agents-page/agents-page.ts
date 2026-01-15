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

  constructor(
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.loadAgents();
  }

  private loadAgents(): void {
    const activeProjectId = this.userService.getActiveProjectId();

    if (!activeProjectId) {
      this.error = 'No active project selected';
      return;
    }

    this.loading = true;
    this.error = null;

    this.agentService.getAgentsByProject(activeProjectId).subscribe({
      next: (agents) => {
        this.agents = agents;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load agents';
        console.error(err);
        this.loading = false;
      }
    });
  }
}
