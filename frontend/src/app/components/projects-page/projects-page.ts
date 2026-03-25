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
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ProjectService } from '../../services/project-service';
import { AgentService } from '../../services/agent-service';
import { Project } from '../../models/project';
import { UserService } from '../../services/user-service';
import { UserDetails } from '../../models/user-details';

@Component({
  selector: 'app-projects-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
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

  // Editing state
  isEditing = false;
  editableProject: Partial<Project> = {};

  // Deletion state
  showDeleteModal = false;
  deleteConfirmationName = '';
  isDeleting = false;

  // User management state
  projectMembers: any[] = [];
  allUsers: UserDetails[] = [];
  showAddUserModal = false;
  newMemberUserId = '';
  newMemberRole: 'owner' | 'viewer' = 'viewer';

  // Section collapse states
  accessDetailsCollapsed = true;
  connectionStringsCollapsed = true; // Default to collapsed for connection strings as they are often long

  ngOnInit() {
    this.userService.userDetails.subscribe(details => {
      if (details && details.active_project_id) {
        const newActiveProjectId = details.active_project_id !== 'None' ? details.active_project_id : null;

        // If active project changed and we already have projects loaded, update selection
        if (this.projects.length > 0 && newActiveProjectId) {
          const projectToSelect = this.projects.find(p => p.id === newActiveProjectId);
          if (projectToSelect && projectToSelect.id !== this.selectedProject?.id) {
            this.selectProject(projectToSelect);
          }
        } else {
          // Initial load or no projects yet
          this.loadProjects();
        }
      } else if (details) {
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
    this.isEditing = false;
    if (project) {
      this.loadSummaryStats(project.id);
      this.loadProjectMembers(project.id);
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

  loadProjectMembers(projectId: string) {
    this.projectService.getMembers(projectId).subscribe({
      next: (members) => {
        this.projectMembers = members;
      },
      error: (err) => {
        console.error('Failed to load project members:', err);
      }
    });
  }

  loadAllUsers() {
    this.userService.getUsers().subscribe({
      next: (users) => {
        this.allUsers = users;
      },
      error: (err) => {
        console.error('Failed to load users:', err);
      }
    });
  }

  editProject() {
    if (this.selectedProject) {
      this.editableProject = { ...this.selectedProject };
      this.isEditing = true;
      this.accessDetailsCollapsed = false;
    }
  }

  cancelEdit() {
    this.isEditing = false;
    this.editableProject = {};
  }

  saveProject() {
    if (this.selectedProject && this.selectedProject.id) {
      this.projectService.update(this.selectedProject.id, this.editableProject).subscribe({
        next: (updated) => {
          this.selectedProject = { ...this.selectedProject, ...updated } as Project;
          this.isEditing = false;
          // Refresh list to show updated name if it changed
          this.loadProjects();
        },
        error: (err) => {
          console.error('Failed to update project:', err);
        }
      });
    }
  }

  confirmDeleteProject() {
    this.deleteConfirmationName = '';
    this.showDeleteModal = true;
  }

  closeDeleteModal() {
    this.showDeleteModal = false;
    this.deleteConfirmationName = '';
  }

  deleteProject() {
    if (this.selectedProject?.id && this.deleteConfirmationName === this.selectedProject.name) {
      this.isDeleting = true;
      this.projectService.delete(this.selectedProject.id).subscribe({
        next: () => {
          this.isDeleting = false;
          this.showDeleteModal = false;
          this.selectedProject = null;
          this.loadProjects();
        },
        error: (err) => {
          this.isDeleting = false;
          console.error('Failed to delete project:', err);
        }
      });
    }
  }

  openAddUserModal() {
    this.loadAllUsers();
    this.newMemberUserId = '';
    this.newMemberRole = 'viewer';
    this.showAddUserModal = true;
  }

  closeAddUserModal() {
    this.showAddUserModal = false;
  }

  addMember() {
    if (this.selectedProject && this.newMemberUserId) {
      this.projectService.addMember(this.selectedProject.id, this.newMemberUserId, this.newMemberRole).subscribe({
        next: () => {
          this.loadProjectMembers(this.selectedProject!.id);
          this.closeAddUserModal();
        },
        error: (err) => {
          console.error('Failed to add member:', err);
        }
      });
    }
  }

  updateMemberRole(userId: string, newRole: string) {
    if (this.selectedProject) {
      this.projectService.updateMember(this.selectedProject.id, userId, newRole).subscribe({
        next: () => {
          this.loadProjectMembers(this.selectedProject!.id);
        },
        error: (err) => {
          console.error('Failed to update member role:', err);
        }
      });
    }
  }

  removeMember(userId: string) {
    if (this.selectedProject && confirm('Remove this user from the project?')) {
      this.projectService.removeMember(this.selectedProject.id, userId).subscribe({
        next: () => {
          this.loadProjectMembers(this.selectedProject!.id);
        },
        error: (err) => {
          console.error('Failed to remove member:', err);
        }
      });
    }
  }

  startNewProject() {
    this.router.navigate(['/project-edit', 'new']);
  }

  getDbUri(type: 'postgresql' | 'asyncpg' | 'jdbc' | 'env', maskPassword = false): string {
    if (!this.selectedProject) return '';
    const p = this.selectedProject;
    const user = p.db_user || 'user';
    const password = p.db_password || (maskPassword ? '••••••••' : 'password');
    const displayPassword = maskPassword ? '••••••••' : password;
    const host = p.db_host || 'localhost';
    const port = p.db_port || 5432;
    const dbName = p.db_name || 'kavalai';

    switch (type) {
      case 'postgresql':
        return `postgresql://${user}:${displayPassword}@${host}:${port}/${dbName}`;
      case 'asyncpg':
        return `postgresql+asyncpg://${user}:${displayPassword}@${host}:${port}/${dbName}`;
      case 'jdbc':
        return `jdbc:postgresql://${host}:${port}/${dbName}?user=${user}&password=${displayPassword}`;
      case 'env':
        return `KAVALAI_DB_URI=postgresql://${user}:${displayPassword}@${host}:${port}/${dbName}`;
      default:
        return '';
    }
  }

  copyToClipboard(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      // Could add a toast notification here if available
      console.log('Copied to clipboard');
    });
  }
}
