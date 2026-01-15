import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { of } from 'rxjs';
import { ProjectsPage } from './projects-page';
import { ProjectService } from '../../services/project-service';
import { UserService } from '../../services/user-service';
import { Project } from '../../models/project';
import { throwError } from 'rxjs';

describe('ProjectsPage', () => {
  let component: ProjectsPage;
  let fixture: ComponentFixture<ProjectsPage>;
  let projectServiceSpy: jasmine.SpyObj<ProjectService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;

  beforeEach(async () => {
    projectServiceSpy = jasmine.createSpyObj('ProjectService', ['getAll', 'create', 'update', 'delete', 'testConnection']);
    userServiceSpy = jasmine.createSpyObj('UserService', ['getIsAdmin', 'setActiveProject']);

    projectServiceSpy.getAll.and.returnValue(of([]));
    userServiceSpy.getIsAdmin.and.returnValue(true);

    await TestBed.configureTestingModule({
      imports: [ProjectsPage, ReactiveFormsModule, CommonModule],
      providers: [
        { provide: ProjectService, useValue: projectServiceSpy },
        { provide: UserService, useValue: userServiceSpy }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(ProjectsPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should load projects on init', () => {
    const mockProjects: Project[] = [{ id: '1', name: 'Project 1' } as Project];
    projectServiceSpy.getAll.and.returnValue(of(mockProjects));

    component.ngOnInit();

    expect(projectServiceSpy.getAll).toHaveBeenCalled();
    expect(component.projects).toEqual(mockProjects);
    expect(component.selectedProject).toEqual(mockProjects[0]);
  });

  it('should toggle edit mode', () => {
    expect(component.isEditing).toBeFalse();
    component.toggleEdit();
    expect(component.isEditing).toBeTrue();
    expect(component.projectForm.enabled).toBeTrue();
  });

  it('should save a new project', () => {
    const newProject = { name: 'New' } as Project;
    component.startNewProject();
    component.projectForm.patchValue({ name: 'New' });
    projectServiceSpy.create.and.returnValue(of({ ...newProject, id: '2' } as Project));
    projectServiceSpy.getAll.and.returnValue(of([{ ...newProject, id: '2' } as Project]));

    component.saveProject();

    expect(projectServiceSpy.create).toHaveBeenCalled();
    expect(userServiceSpy.setActiveProject).toHaveBeenCalledWith('2');
  });

  it('should test connection', () => {
    component.selectedProject = { id: '1' } as Project;
    const mockRes = { status: 'success', message: 'Connected' };
    projectServiceSpy.testConnection.and.returnValue(of(mockRes));

    component.testConnection();

    expect(projectServiceSpy.testConnection).toHaveBeenCalledWith('1');
    expect(component.connectionStatus).toEqual(mockRes);
  });

  it('should update an existing project', () => {
    const existingProject = { id: '1', name: 'Existing' } as Project;
    component.selectedProject = existingProject;
    component.isEditing = true;
    component.projectForm.patchValue({ name: 'Updated' });
    projectServiceSpy.update.and.returnValue(of({ ...existingProject, name: 'Updated' }));
    projectServiceSpy.getAll.and.returnValue(of([{ ...existingProject, name: 'Updated' }]));

    component.saveProject();

    expect(projectServiceSpy.update).toHaveBeenCalledWith('1', jasmine.any(Object));
    expect(userServiceSpy.setActiveProject).toHaveBeenCalledWith('1');
  });

  it('should delete a project after confirmation', () => {
    const existingProject = { id: '1', name: 'Existing' } as Project;
    component.selectedProject = existingProject;
    spyOn(window, 'confirm').and.returnValue(true);
    projectServiceSpy.delete.and.returnValue(of(undefined));
    projectServiceSpy.getAll.and.returnValue(of([]));

    component.deleteProject();

    expect(projectServiceSpy.delete).toHaveBeenCalledWith('1');
    expect(component.selectedProject).toBeNull();
  });

  it('should not delete a project if not confirmed', () => {
    const existingProject = { id: '1', name: 'Existing' } as Project;
    component.selectedProject = existingProject;
    spyOn(window, 'confirm').and.returnValue(false);

    component.deleteProject();

    expect(projectServiceSpy.delete).not.toHaveBeenCalled();
  });

  it('should handle project selection from event', () => {
    const mockProjects: Project[] = [
      { id: '1', name: 'P1' } as Project,
      { id: '2', name: 'P2' } as Project
    ];
    component.projects = mockProjects;
    const event = { target: { value: '2' } } as any;

    component.onProjectSelect(event);


    expect(component.selectedProject?.id).toBe('2');
    expect(userServiceSpy.setActiveProject).toHaveBeenCalledWith('2');
  });

  it('should handle test connection error', () => {
    component.selectedProject = { id: '1' } as Project;
    const errorResponse = { error: { message: 'Failed' } };
    projectServiceSpy.testConnection.and.returnValue(throwError(() => errorResponse));

    component.testConnection();

    expect(component.connectionStatus?.status).toBe('error');
    expect(component.connectionStatus?.message).toBe('Failed');
  });

  it('should handle test connection error without message', () => {
    component.selectedProject = { id: '1' } as Project;
    projectServiceSpy.testConnection.and.returnValue(throwError(() => new Error('Generic error')));

    component.testConnection();

    expect(component.connectionStatus?.status).toBe('error');
    expect(component.connectionStatus?.message).toBe('Generic error');
  });

  it('should not save if form is invalid', () => {
    component.startNewProject();
    component.projectForm.patchValue({ name: '' }); // name is required
    component.saveProject();
    expect(projectServiceSpy.create).not.toHaveBeenCalled();
  });

  it('should revert changes when cancelling edit', () => {
    const existingProject = { id: '1', name: 'Original' } as Project;
    component.selectProject(existingProject);
    component.toggleEdit();
    component.projectForm.patchValue({ name: 'Changed' });

    component.toggleEdit(); // toggle back to false

    expect(component.projectForm.get('name')?.value).toBe('Original');
  });
});
