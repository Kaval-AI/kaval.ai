import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { of } from 'rxjs';
import { ProjectsPage } from './projects-page';
import { ProjectService } from '../../services/project-service';
import { UserService } from '../../services/user-service';
import { Project } from '../../models/project';

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
});
