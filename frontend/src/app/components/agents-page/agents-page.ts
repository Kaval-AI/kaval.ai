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
import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Agent } from '../../models/agent';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions, Chart, registerables } from 'chart.js';
import { JsonTreeComponent } from '../json-tree/json-tree';
import { NavigationService } from '../../services/navigation-service';

Chart.register(...registerables);

@Component({
  selector: 'app-agents-page',
  standalone: true,
  imports: [CommonModule, BaseChartDirective, JsonTreeComponent],
  templateUrl: './agents-page.html',
  styleUrl: './agents-page.css'
})
export class AgentsPage implements OnInit {
  private agentService = inject(AgentService);
  private userService = inject(UserService);
  private router = inject(Router);
  private navigationService = inject(NavigationService);

  agents: Agent[] = [];
  loading: boolean = false;
  error: string | null = null;
  selectedAgent: Agent | null = null;
  activeProjectId: string | null = null;

  showModal: boolean = false;
  modalTitle: string = '';
  modalData: any = null;

  summaryStats: any = null;
  dailyStats: any = null;

  // Chart state
  public activityChartData: ChartConfiguration<'line'>['data'] = {
    datasets: [],
    labels: []
  };
  public tokensChartData: ChartConfiguration<'line'>['data'] = {
    datasets: [],
    labels: []
  };
  public durationsChartData: ChartConfiguration<'line'>['data'] = {
    datasets: [],
    labels: []
  };
  public runtimesChartData: ChartConfiguration<'line'>['data'] = {
    datasets: [],
    labels: []
  };
  public chartOptions: ChartOptions<'line'> = {
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
        display: true,
        position: 'top',
        labels: {
          boxWidth: 10,
          usePointStyle: true,
          pointStyle: 'circle'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    }
  };

  ngOnInit(): void {
    this.navigationService.setTitle('Agents');
    this.userService.userDetails.subscribe(user => {
      if (user) {
        const newProjectId = (user.active_project_id && user.active_project_id !== 'None') ? user.active_project_id : null;
        if (newProjectId !== this.activeProjectId) {
          this.activeProjectId = newProjectId;
          this.selectedAgent = null;
          this.summaryStats = null;
          this.dailyStats = null;
          this.loadAgents();
        } else if (newProjectId === null) {
          this.loadAgents();
        }
      }
    });
  }

  private loadStats(agentId: string): void {
    if (!this.activeProjectId) return;

    this.agentService.getSummaryStats(this.activeProjectId, agentId).subscribe({
      next: (stats) => {
        this.summaryStats = stats;
      },
      error: (err) => {
        console.error('Failed to load summary stats', err);
      }
    });

    this.agentService.getDailyStats(this.activeProjectId, 7, agentId).subscribe({
      next: (stats) => {
        this.dailyStats = stats;
        this.prepareChartData();
      },
      error: (err) => {
        console.error('Failed to load daily stats', err);
        this.dailyStats = null;
      }
    });
  }

  private prepareChartData(): void {
    if (!this.dailyStats) return;

    const labels = this.dailyStats.sessions.map((d: any) => {
      const date = new Date(d.date);
      const day = String(date.getDate()).padStart(2, '0');
      const month = String(date.getMonth() + 1).padStart(2, '0');
      return `${day}-${month}`;
    });

    const colors = {
      sessions: '#acc12f',
      messages: '#1b998b',
      tasks: '#b185a7',
      runs: '#6d9999',
      input: '#e67f0d',
      output: '#82204a',
      embedding: '#809537',
      llm_duration: '#4a90e2',
      embedding_duration: '#50e3c2'
    };

    const palette = [
      '#acc12f', '#1b998b', '#b185a7', '#6d9999', '#e67f0d', '#82204a', '#809537',
      '#4a90e2', '#50e3c2', '#f5a623', '#bd10e0', '#9013fe'
    ];

    // Activity Chart
    this.activityChartData = {
      labels: labels,
      datasets: [
        {
          data: this.dailyStats.sessions.map((d: any) => d.count),
          label: 'Sessions',
          borderColor: colors.sessions,
          backgroundColor: colors.sessions,
          fill: false,
          tension: 0.1,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        },
        {
          data: this.dailyStats.messages.map((d: any) => d.count),
          label: 'Messages',
          borderColor: colors.messages,
          backgroundColor: colors.messages,
          fill: false,
          tension: 0.1,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        },
        {
          data: this.dailyStats.tasks.map((d: any) => d.count),
          label: 'Tasks',
          borderColor: colors.tasks,
          backgroundColor: colors.tasks,
          fill: false,
          tension: 0.1,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        },
        {
          data: labels.map((_: any, i: number) => {
            let total = 0;
            for (const agent of Object.values(this.dailyStats.runs) as any[]) {
              total += agent[i]?.count || 0;
            }
            return total;
          }),
          label: 'Workflow Runs',
          borderColor: colors.runs,
          backgroundColor: colors.runs,
          fill: false,
          tension: 0.1,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        }
      ]
    };

    // Tokens Chart
    this.tokensChartData = {
      labels: labels,
      datasets: [
        {
          data: labels.map((_: any, i: number) => {
            let total = 0;
            for (const model of Object.values(this.dailyStats.llm) as any[]) {
              total += model[i]?.prompt_tokens || 0;
            }
            return total;
          }),
          label: 'Input Tokens',
          borderColor: colors.input,
          backgroundColor: colors.input,
          fill: false,
          tension: 0.1,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        },
        {
          data: labels.map((_: any, i: number) => {
            let total = 0;
            for (const model of Object.values(this.dailyStats.llm) as any[]) {
              total += model[i]?.completion_tokens || 0;
            }
            return total;
          }),
          label: 'Output Tokens',
          borderColor: colors.output,
          backgroundColor: colors.output,
          fill: false,
          tension: 0.1,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        },
        {
          data: labels.map((_: any, i: number) => {
            let total = 0;
            for (const model of Object.values(this.dailyStats.embedding) as any[]) {
              total += model[i]?.total_tokens || 0;
            }
            return total;
          }),
          label: 'Embedding Tokens',
          borderColor: colors.embedding,
          backgroundColor: colors.embedding,
          fill: false,
          tension: 0.1,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        }
      ]
    };

    // Durations Chart
    this.durationsChartData = {
      labels: labels,
      datasets: [
        {
          data: labels.map((_: any, i: number) => {
            let totalDuration = 0;
            let totalCount = 0;
            for (const model of Object.values(this.dailyStats.llm) as any[]) {
              totalDuration += model[i]?.duration_seconds || 0;
              totalCount += model[i]?.count || 0;
            }
            return totalCount > 0 ? totalDuration / totalCount : 0;
          }),
          label: 'Avg LLM Duration (s)',
          borderColor: colors.llm_duration,
          backgroundColor: colors.llm_duration,
          fill: false,
          tension: 0.1,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        },
        {
          data: labels.map((_: any, i: number) => {
            let totalDuration = 0;
            let totalCount = 0;
            for (const model of Object.values(this.dailyStats.embedding) as any[]) {
              totalDuration += model[i]?.duration_seconds || 0;
              totalCount += model[i]?.count || 0;
            }
            return totalCount > 0 ? totalDuration / totalCount : 0;
          }),
          label: 'Avg Embedding Duration (s)',
          borderColor: colors.embedding_duration,
          backgroundColor: colors.embedding_duration,
          fill: false,
          tension: 0.1,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        }
      ]
    };

    // Runtimes Chart
    this.runtimesChartData = {
      labels: labels,
      datasets: Object.keys(this.dailyStats.runs).map((agentName, idx) => ({
        data: this.dailyStats.runs[agentName].map((d: any) => d.duration_seconds || 0),
        label: agentName,
        borderColor: palette[idx % palette.length],
        backgroundColor: palette[idx % palette.length],
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      }))
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

  openModal(type: 'workflow' | 'input' | 'output'): void {
    if (!this.selectedAgent) return;

    this.showModal = true;
    if (type === 'workflow') {
      this.modalTitle = 'Workflow JSON';
      this.modalData = this.selectedAgent.workflow;
    } else if (type === 'input') {
      this.modalTitle = 'Input Schema';
      this.modalData = this.selectedAgent.input_schema;
    } else if (type === 'output') {
      this.modalTitle = 'Output Schema';
      this.modalData = this.selectedAgent.output_schema;
    }
  }

  closeModal(): void {
    this.showModal = false;
    this.modalData = null;
  }
}
