import { Component, OnInit, inject } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ProjectService } from '../../services/project-service';
import { Project } from '../../models/project';
import { UserService } from '../../services/user-service';

@Component({
  selector: 'app-projects-page',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule],
  templateUrl: './projects-page.html',
  styleUrl: './projects-page.css',
})
export class ProjectsPage implements OnInit {
  private projectService = inject(ProjectService);
  private userService = inject(UserService);
  private fb = inject(FormBuilder);

  projects: Project[] = [];
  selectedProject: Project | null = null;
  isEditing = false;

  // Updated Form Controls to match the new DB columns
  projectForm = this.fb.group({
    name: ['', [Validators.required]],
    description: ['', []],
    db_host: ['', []],
    db_port: [5432, [Validators.min(1), Validators.max(65535)]],
    db_user: ['', []],
    db_password: ['', []],
    db_name: ['', []],
    db_schema: ['public', []]
  });

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
      // Revert changes if cancelling
      this.projectForm.patchValue(this.selectedProject);
    }
  }

  selectProject(project: Project | null) {
    this.selectedProject = project;
    this.isEditing = false;

    if (project) {
      // patchValue automatically maps matching keys from 'project' to form controls
      this.projectForm.patchValue(project);
    } else {
      this.projectForm.reset({
        db_port: 5432,
        db_schema: 'public'
      });
    }
    this.updateFormState();
  }

  saveProject() {
    if (this.projectForm.invalid) return;

    const formValue = this.projectForm.value as Partial<Project>;

    if (this.selectedProject?.id) {
      this.projectService.update(this.selectedProject.id, formValue).subscribe(() => {
        this.isEditing = false;
        this.loadProjects();
        if (this.selectedProject?.id) {
          this.userService.setActiveProject(this.selectedProject.id);
        }
      });
    } else {
      this.projectService.create(formValue).subscribe((newProj: Project) => {
        this.isEditing = false;
        this.loadProjects();
        this.selectProject(newProj)
        this.userService.setActiveProject(newProj.id);
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
    this.projectForm.reset({
      db_port: 5432,
      db_schema: 'public'
    });
    this.updateFormState();
  }
}
