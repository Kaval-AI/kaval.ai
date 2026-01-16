import { Component, OnInit, inject } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { ProjectService } from '../../services/project-service';
import { Project } from '../../models/project';
import { UserService } from '../../services/user-service';

@Component({
  selector: 'app-project-edit-page',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule],
  templateUrl: './project-edit-page.html',
  styleUrl: './project-edit-page.css',
})
export class ProjectEditPage implements OnInit {
  private projectService = inject(ProjectService);
  private userService = inject(UserService);
  private fb = inject(FormBuilder);
  private route = inject(ActivatedRoute);
  private router = inject(Router);

  projectId: string | null = null;
  project: Project | null = null;
  connectionStatus: { status: string; message?: string } | null = null;
  isTestingConnection = false;

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
    this.projectId = this.route.snapshot.paramMap.get('id');
    if (this.projectId && this.projectId !== 'new') {
      this.loadProject(this.projectId);
    }
  }

  loadProject(id: string) {
    this.projectService.getAll().subscribe((projects) => {
      const p = projects.find(proj => proj.id === id);
      if (p) {
        this.project = p;
        this.projectForm.patchValue(p);
      }
    });
  }

  saveProject() {
    if (this.projectForm.invalid) return;

    const formValue = this.projectForm.value as Partial<Project>;

    if (this.projectId && this.projectId !== 'new') {
      this.projectService.update(this.projectId, formValue).subscribe(() => {
        this.router.navigate(['/']);
      });
    } else {
      this.projectService.create(formValue).subscribe((newProj: Project) => {
        this.userService.setActiveProject(newProj.id);
        this.router.navigate(['/']);
      });
    }
  }

  cancel() {
    this.router.navigate(['/']);
  }

  testConnection() {
    if (!this.projectId || this.projectId === 'new') return;

    this.isTestingConnection = true;
    this.connectionStatus = null;

    this.projectService.testConnection(this.projectId).subscribe({
      next: (res) => {
        this.connectionStatus = res;
        this.isTestingConnection = false;
      },
      error: (err) => {
        this.connectionStatus = {
          status: 'error',
          message: err.error?.message || err.message || 'Unknown error'
        };
        this.isTestingConnection = false;
      }
    });
  }
}
