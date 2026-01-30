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
import { Router } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Agent } from '../../models/agent';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions, ChartType, Chart, registerables } from 'chart.js';
import { JsonTreeComponent } from '../json-tree/json-tree';

Chart.register(...registerables);

@Component({
  selector: 'app-agents-page',
  standalone: true,
  imports: [CommonModule, BaseChartDirective, JsonTreeComponent],
  templateUrl: './agents-page.html',
  styleUrl: './agents-page.css'
})
export class AgentsPage implements OnInit {
  agents: Agent[] = [];
  loading: boolean = false;
  error: string | null = null;
  selectedAgent: Agent | null = null;
  activeProjectId: string | null = null;

  stats: any = null;
  public lineChartData: ChartConfiguration<'line'>['data'] = {
    datasets: [],
    labels: []
  };
  public lineChartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1
        }
      }
    },
    plugins: {
      legend: {
        display: true
      }
    }
  };
  public lineChartLegend = true;

  constructor(
    private agentService: AgentService,
    private userService: UserService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.activeProjectId = this.userService.getActiveProjectId();
    this.loadAgents();
  }

  private loadStats(agentId?: string): void {
    if (!this.activeProjectId) return;

    this.agentService.getAgentStats(this.activeProjectId, agentId).subscribe({
      next: (stats) => {
        this.stats = stats;
        this.prepareChartData();
      },
      error: (err) => {
        console.error('Failed to load stats', err);
      }
    });
  }

  private prepareChartData(): void {
    if (!this.stats) return;

    const labels = this.stats.runs.map((d: any) => {
      const date = new Date(d.date);
      const day = String(date.getDate()).padStart(2, '0');
      const month = String(date.getMonth() + 1).padStart(2, '0');
      return `${day}-${month}`;
    });

    this.lineChartData = {
      labels: labels,
      datasets: [
        {
          data: this.stats.runs.map((d: any) => d.count),
          label: 'Runs',
          borderColor: '#42A5F5',
          backgroundColor: '#42A5F5',
          fill: false,
          tension: 0,
          borderWidth: 3,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          data: this.stats.sessions.map((d: any) => d.count),
          label: 'Sessions',
          borderColor: '#FFA726',
          backgroundColor: '#FFA726',
          fill: false,
          tension: 0,
          borderWidth: 3,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          data: this.stats.messages.map((d: any) => d.count),
          label: 'Messages',
          borderColor: '#66BB6A',
          backgroundColor: '#66BB6A',
          fill: false,
          tension: 0,
          borderWidth: 3,
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    };
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
          this.selectAgent(this.agents[0]);
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
    this.loadStats(agent.id);
  }

  getSvgUrl(agentId: string): string {
    if (!this.activeProjectId) return '';
    return this.agentService.getAgentSvgUrl(this.activeProjectId, agentId);
  }

  goToConversations(): void {
    if (this.selectedAgent) {
      this.router.navigate(['/conversations'], { queryParams: { agentId: this.selectedAgent.id } });
    }
  }
}
