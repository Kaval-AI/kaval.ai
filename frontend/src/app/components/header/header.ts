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
import { Component, inject, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { UserInfo } from '../user-info/user-info';
import { NavigationService } from '../../services/navigation-service';
import { ProjectService } from '../../services/project-service';
import { UserService } from '../../services/user-service';
import { Project } from '../../models/project';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [UserInfo, CommonModule],
  templateUrl: './header.html',
  styleUrl: './header.css',
})
export class Header implements OnInit {
  private navigationService = inject(NavigationService);
  private projectService = inject(ProjectService);
  private userService = inject(UserService);
  private router = inject(Router);

  title = this.navigationService.title;
  projects: Project[] = [];
  activeProjectId: string | null = null;

  ngOnInit(): void {
    this.userService.userDetails.subscribe(details => {
      if (details) {
        // Set active project first so template can reflect it while projects load
        this.activeProjectId = details.active_project_id && details.active_project_id !== 'None' ? details.active_project_id : null;
        this.loadProjects();
      } else {
        this.projects = [];
        this.activeProjectId = null;
      }
    });
  }

  loadProjects(): void {
    this.projectService.getAll().subscribe((data: Project[]) => {
      this.projects = data;

      const activeProjectId = this.userService.getActiveProjectId();
      if (activeProjectId) {
        // Ensure the select reflects the currently active project from the service
        this.activeProjectId = activeProjectId;
      } else if (this.projects.length > 0) {
        // If none is set, default to the first and update both service and local state
        const firstId = this.projects[0].id;
        this.userService.setActiveProject(firstId);
        this.activeProjectId = firstId;
      }
    });
  }

  onProjectSelect(event: Event): void {
    const target = event.target as HTMLSelectElement;
    const projectId = target.value;
    if (projectId) {
      this.userService.setActiveProject(projectId);
      this.router.navigateByUrl('/', { skipLocationChange: true }).then(() => {
        this.router.navigate([this.router.url]);
      });
    }
  }

  startNewProject() {
    this.router.navigate(['/project-edit', 'new']);
  }
}
