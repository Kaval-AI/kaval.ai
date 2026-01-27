import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { LLMConfig, LLMEmbeddingConfig } from '../../models/llm-config';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions, Chart, registerables } from 'chart.js';
import { forkJoin } from 'rxjs';

Chart.register(...registerables);

@Component({
  selector: 'app-configs-page',
  standalone: true,
  imports: [CommonModule, BaseChartDirective, RouterLink],
  templateUrl: './configs-page.html',
  styleUrl: './configs-page.css',
})
export class ConfigsPage implements OnInit {
  configs: LLMConfig[] = [];
  embeddingConfigs: LLMEmbeddingConfig[] = [];
  loading: boolean = false;
  error: string | null = null;
  activeProjectId: string | null = null;
  stats: any = null;

  public callsChartData: ChartConfiguration<'line'>['data'] = { datasets: [], labels: [] };
  public costChartData: ChartConfiguration<'line'>['data'] = { datasets: [], labels: [] };
  public tokensChartData: ChartConfiguration<'line'>['data'] = { datasets: [], labels: [] };
  public durationChartData: ChartConfiguration<'line'>['data'] = { datasets: [], labels: [] };

  public chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
      }
    },
    plugins: {
      legend: { display: true, position: 'bottom' }
    }
  };

  public callsChartOptions: ChartOptions<'line'> = {
    ...this.chartOptions,
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1
        }
      }
    }
  };

  private colors = [
    '#42A5F5', '#FFA726', '#66BB6A', '#EF5350', '#AB47BC', '#26C6DA', '#FFCA28'
  ];

  constructor(
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.activeProjectId = this.userService.getActiveProjectId();
    this.loadConfigs();
    this.loadStats();
  }

  loadConfigs(): void {
    if (!this.activeProjectId) {
      this.error = 'No active project selected';
      return;
    }

    this.loading = true;
    this.error = null;

    forkJoin({
      llm: this.agentService.getLLMConfigs(this.activeProjectId),
      embedding: this.agentService.getEmbeddingConfigs(this.activeProjectId)
    }).subscribe({
      next: (res) => {
        this.configs = res.llm;
        this.embeddingConfigs = res.embedding;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load configurations';
        console.error(err);
        this.loading = false;
      }
    });
  }

  loadStats(): void {
    if (!this.activeProjectId) return;

    this.agentService.getAgentStats(this.activeProjectId).subscribe({
      next: (stats) => {
        this.stats = stats;
        this.prepareCharts();
      },
      error: (err) => {
        console.error('Failed to load LLM stats', err);
      }
    });
  }

  private prepareCharts(): void {
    if (!this.stats || !this.stats.llm) return;

    const profileNames = Object.keys(this.stats.llm);
    if (profileNames.length === 0) {
      // Clear charts if no stats
      this.callsChartData = { datasets: [], labels: [] };
      this.costChartData = { datasets: [], labels: [] };
      this.tokensChartData = { datasets: [], labels: [] };
      this.durationChartData = { datasets: [], labels: [] };
      return;
    }

    const firstProfile = this.stats.llm[profileNames[0]];
    const labels = firstProfile.map((d: any) => {
      const date = new Date(d.date);
      const day = String(date.getDate()).padStart(2, '0');
      const month = String(date.getMonth() + 1).padStart(2, '0');
      return `${day}-${month}`;
    });

    const callsDatasets: any[] = [];
    const costDatasets: any[] = [];
    const tokensDatasets: any[] = [];
    const durationDatasets: any[] = [];

    profileNames.forEach((name, index) => {
      const profileStats = this.stats.llm[name];
      const color = this.colors[index % this.colors.length];

      callsDatasets.push({
        data: profileStats.map((d: any) => d.count),
        label: name,
        borderColor: color,
        backgroundColor: color,
        fill: false,
        tension: 0,
        borderWidth: 3,
        pointRadius: 4,
        pointHoverRadius: 6
      });

      costDatasets.push({
        data: profileStats.map((d: any) => d.cost),
        label: name,
        borderColor: color,
        backgroundColor: color,
        fill: false,
        tension: 0,
        borderWidth: 3,
        pointRadius: 4,
        pointHoverRadius: 6
      });

      durationDatasets.push({
        data: profileStats.map((d: any) => d.duration_seconds),
        label: name,
        borderColor: color,
        backgroundColor: color,
        fill: false,
        tension: 0,
        borderWidth: 3,
        pointRadius: 4,
        pointHoverRadius: 6
      });

      tokensDatasets.push({
        data: profileStats.map((d: any) => d.prompt_tokens),
        label: `${name} (Input)`,
        borderColor: color,
        backgroundColor: color,
        borderDash: [5, 5],
        fill: false,
        tension: 0,
        borderWidth: 3,
        pointRadius: 4,
        pointHoverRadius: 6
      });

      tokensDatasets.push({
        data: profileStats.map((d: any) => d.completion_tokens),
        label: `${name} (Output)`,
        borderColor: color,
        backgroundColor: color,
        fill: false,
        tension: 0,
        borderWidth: 3,
        pointRadius: 4,
        pointHoverRadius: 6
      });
    });

    this.callsChartData = { labels, datasets: callsDatasets };
    this.costChartData = { labels, datasets: costDatasets };
    this.durationChartData = { labels, datasets: durationDatasets };
    this.tokensChartData = { labels, datasets: tokensDatasets };
  }

  formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }
}
