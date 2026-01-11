import { Component, OnInit, inject } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ProjectService } from '../../services/project-service';
import { Project } from '../../models/project';
import { AuthService } from '../../services/auth-service';

@Component({
  selector: 'app-projects-page',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule],
  templateUrl: './projects-page.html',
  styleUrl: './projects-page.css',
})
export class ProjectsPage implements OnInit {
  private projectService = inject(ProjectService);
  private authService = inject(AuthService);
  private fb = inject(FormBuilder);

  projects: Project[] = [];
  selectedProject: Project | null = null;
  isEditing = false;

  projectForm = this.fb.group({
    name: ['', [Validators.required]],
    description: ['', []]
  });

  ngOnInit() {
    this.loadProjects();
  }

  loadProjects() {
    console.log("Loading projects")
    this.projectService.getAll().subscribe((data: Project[]) => {
      this.projects = data;
      this.selectProject(this.projects.length > 0 ? this.projects[0] : null);
    });
  }

  getIsAdmin() {
    return this.authService.getIsAdmin();
  }

  onProjectSelect(event: Event) {
    const target = event.target as HTMLSelectElement;
    const project = this.projects.find(p => p.id === target.value);
    if (project) {
      this.selectProject(project);
    }
  }

  private updateFormState() {
    if (this.isEditing) {
      this.projectForm.enable();
    } else {
      this.projectForm.disable();
    }
  }

  toggleEdit() {
    this.isEditing = !this.isEditing;
    this.updateFormState();

    if (!this.isEditing && this.selectedProject) {
      this.projectForm.patchValue(this.selectedProject);
    }
  }

  selectProject(project: Project | null) {
    this.selectedProject = project;
    this.isEditing = false; // Reset to view mode on selection

    if (project) {
      this.projectForm.patchValue(project);
    } else {
      this.projectForm.reset();
    }
    this.updateFormState();
  }

  saveProject() {
    if (this.projectForm.invalid) return;
    const formValue = this.projectForm.value as Partial<Project>;
    this.isEditing = false;
    if (this.selectedProject?.id) {
      this.projectService.update(this.selectedProject.id, formValue).subscribe(() => {
        this.loadProjects();
      });
    } else {
      this.projectService.create(formValue).subscribe((newProj: Project) => {
        this.loadProjects();
        this.selectProject(newProj);
      });
    }
  }

  deleteProject() {
    if (this.selectedProject?.id && confirm('Delete this project?')) {
      this.projectService.delete(this.selectedProject.id).subscribe(() => {
        this.selectedProject = null;
        this.projectForm.reset();
        this.loadProjects();
      });
    }
  }

  startNewProject() {
    this.selectedProject = null;
    this.isEditing = true;
    this.projectForm.reset();
    this.updateFormState();
  }
}
