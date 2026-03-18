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
import { Task } from '../../models/task';
import { TasksList } from '../tasks-list/tasks-list';

@Component({
  selector: 'app-run-tasks-page',
  standalone: true,
  imports: [CommonModule, RouterModule, TasksList],
  templateUrl: './run-tasks-page.html',
  styleUrl: './run-tasks-page.css',
})
export class RunTasksPage implements OnInit {
  sessionId: string | null = null;
  runId: string | null = null;
  projectId: string | null = null;
  tasks: Task[] = [];
  loading: boolean = false;
  error: string | null = null;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.sessionId = params.get('sessionId');
      this.runId = params.get('runId');
      if (this.sessionId && this.runId) {
        this.tryLoad();
      }
    });

    this.userService.userDetails.subscribe((user) => {
      if (user && user.active_project_id) {
        const newProjectId = user.active_project_id !== 'None' ? user.active_project_id : null;
        if (newProjectId !== this.projectId) {
          this.projectId = newProjectId;
          this.tryLoad();
        }
      }
    });
  }

  private tryLoad(): void {
    if (!this.projectId || !this.sessionId || !this.runId) return;
    this.loadTasks();
  }

  private loadTasks(): void {
    if (!this.projectId || !this.sessionId || !this.runId) return;
    this.loading = true;
    this.error = null;

    this.agentService.getSessionDetails(this.projectId, this.sessionId).subscribe({
      next: (details) => {
        this.tasks = details.tasks.filter(t => t.run_id === this.runId);
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load tasks';
        console.error(err);
        this.loading = false;
      }
    });
  }

  goBack(): void {
    if (this.sessionId) {
      this.router.navigate(['/conversations', this.sessionId]);
    } else {
      this.router.navigate(['/conversations']);
    }
  }
}
