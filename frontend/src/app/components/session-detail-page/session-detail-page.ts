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
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { ChatMessage } from '../../models/chat-message';
import { Run } from '../../models/run';
import { Task } from '../../models/task';
import { JsonTreeComponent } from '../json-tree/json-tree';
import { NavigationService } from '../../services/navigation-service';

interface RunBlock {
  run: Run;
  messages: ChatMessage[];
  tasks: Task[];
}

@Component({
  selector: 'app-session-detail-page',
  standalone: true,
  imports: [CommonModule, RouterModule, JsonTreeComponent],
  templateUrl: './session-detail-page.html',
  styleUrl: './session-detail-page.css',
})
export class SessionDetailPage implements OnInit {
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private agentService = inject(AgentService);
  private userService = inject(UserService);
  private navigationService = inject(NavigationService);

  sessionId: string | null = null;
  projectId: string | null = null;
  runBlocks: RunBlock[] = [];
  unassignedMessages: ChatMessage[] = [];
  loading: boolean = false;
  error: string | null = null;

  // Modal state
  showModal: boolean = false;
  modalTitle: string = '';
  modalData: any = null;
  modalType: 'json' | 'tasks' = 'json';
  modalRunId: string = '';

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.sessionId = params.get('sessionId');
      this.navigationService.setBreadcrumbs([
        { label: 'Conversations', link: '/conversations' },
        { label: this.sessionId || 'Session Details' }
      ]);
    });

    this.userService.userDetails.subscribe((user) => {
      if (user && user.active_project_id) {
        const newProjectId =
          user.active_project_id !== 'None' ? user.active_project_id : null;
        if (newProjectId !== this.projectId) {
          this.projectId = newProjectId;
          if (this.sessionId && this.projectId) {
            this.loadSessionDetails();
          }
        }
      } else if (this.sessionId) {
        this.error = 'Invalid session or project';
      }
    });
  }

  loadSessionDetails(): void {
    if (!this.projectId || !this.sessionId) return;

    this.loading = true;
    this.error = null;

    this.agentService.getSessionDetails(this.projectId, this.sessionId).subscribe({
      next: (details) => {
        const runMap = new Map<string, RunBlock>();
        details.runs.forEach((run) => {
          runMap.set(run.id, { run, messages: [], tasks: [] });
        });

        this.unassignedMessages = [];
        details.messages.forEach((msg) => {
          if (msg.run_id && runMap.has(msg.run_id)) {
            runMap.get(msg.run_id)!.messages.push(msg);
          } else {
            this.unassignedMessages.push(msg);
          }
        });

        details.tasks.forEach((task) => {
          if (runMap.has(task.run_id)) {
            runMap.get(task.run_id)!.tasks.push(task);
          }
        });

        this.runBlocks = Array.from(runMap.values());
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load session details';
        console.error(err);
        this.loading = false;
      },
    });
  }

  openJsonModal(title: string, data: any): void {
    this.modalTitle = title;
    this.modalData = data;
    this.modalType = 'json';
    this.showModal = true;
  }

  openTasksPage(run: Run): void {
    if (!this.sessionId) return;
    this.router.navigate(['/conversations', this.sessionId, 'runs', run.id, 'tasks']);
  }

  closeModal(): void {
    this.showModal = false;
    this.modalData = null;
  }

  formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }

  getTaskNames(tasks: Task[]): string {
    return tasks
      .map((t) => t.name || t.id)
      .filter((name, index, self) => self.indexOf(name) === index)
      .join(', ');
  }

  goBack(): void {
    this.router.navigate(['/conversations']);
  }
}
