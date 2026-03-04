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
import { ProjectService } from '../../services/project-service';
import { AgentService } from '../../services/agent-service';
import { Project } from '../../models/project';
import { UserService } from '../../services/user-service';

@Component({
  selector: 'app-projects-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './projects-page.html',
  styleUrl: './projects-page.css',
})
export class ProjectsPage implements OnInit {
  private projectService = inject(ProjectService);
  private agentService = inject(AgentService);
  private userService = inject(UserService);
  private router = inject(Router);

  projects: Project[] = [];
  selectedProject: Project | null = null;
  summaryStats: any = null;
  dbConnectionError: string | null = null;

  ngOnInit() {
    this.userService.userDetails.subscribe(details => {
      if (details) {
        this.loadProjects();
      }
    });
  }

  loadProjects() {
    this.projectService.getAll().subscribe({
      next: (data: Project[]) => {
        this.projects = data;
        this.dbConnectionError = null;
        // If we already have a selected project, keep it selected (and updated)
        // otherwise, grab the active one from user service, or the first one.
        const activeProjectId = this.userService.getActiveProjectId();
        const toSelect = this.projects.find(p => p.id === this.selectedProject?.id) ||
                         (activeProjectId ? this.projects.find(p => p.id === activeProjectId) : null) ||
                         (this.projects.length > 0 ? this.projects[0] : null);

        this.selectProject(toSelect);
      },
      error: (err) => {
        if (err.status === 503) {
          this.dbConnectionError = err.error?.detail || 'Backoffice database is not connected.';
          this.projects = [];
        } else {
          console.error('Failed to load projects:', err);
        }
      }
    });
  }

  getIsAdmin() {
    return this.userService.getIsAdmin();
  }

  selectProject(project: Project | null) {
    this.selectedProject = project;
    this.summaryStats = null;
    this.dbConnectionError = null;
    if (project) {
      this.loadSummaryStats(project.id);
    }
  }

  loadSummaryStats(projectId: string) {
    this.agentService.getSummaryStats(projectId).subscribe({
      next: (stats) => {
        this.summaryStats = stats;
        this.dbConnectionError = null;
      },
      error: (err) => {
        if (err.status === 503) {
          this.dbConnectionError = err.error?.detail || 'Database is not connected.';
        } else {
          console.error('Failed to load summary stats:', err);
        }
      }
    });
  }

  editProject() {
    if (this.selectedProject?.id) {
      this.router.navigate(['/project-edit', this.selectedProject.id]);
    }
  }

  deleteProject() {
    if (this.selectedProject?.id && confirm('Delete this project?')) {
      this.projectService.delete(this.selectedProject.id).subscribe(() => {
        this.selectedProject = null;
        this.loadProjects();
      });
    }
  }

  startNewProject() {
    this.router.navigate(['/project-edit', 'new']);
  }
}
