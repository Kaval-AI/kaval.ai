import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { ProjectService } from '../../services/project-service';
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
  private userService = inject(UserService);
  private router = inject(Router);

  projects: Project[] = [];
  selectedProject: Project | null = null;

  ngOnInit() {
    this.loadProjects();
  }

  loadProjects() {
    this.projectService.getAll().subscribe((data: Project[]) => {
      this.projects = data;
      // If we already have a selected project, keep it selected (and updated)
      // otherwise, grab the first one.
      const toSelect = this.projects.find(p => p.id === this.selectedProject?.id) ||
                       (this.projects.length > 0 ? this.projects[0] : null);
      this.selectProject(toSelect);
    });
  }

  getIsAdmin() {
    return this.userService.getIsAdmin();
  }

  onProjectSelect(event: Event) {
    const target = event.target as HTMLSelectElement;
    const project = this.projects.find(p => p.id === target.value);
    if (project) {
      this.selectProject(project);
      this.userService.setActiveProject(project.id);
    }
  }

  selectProject(project: Project | null) {
    this.selectedProject = project;
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
