import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CommonModule } from '@angular/common';
import { of } from 'rxjs';
import { ProjectsPage } from './projects-page';
import { ProjectService } from '../../services/project-service';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Project } from '../../models/project';
import { Router } from '@angular/router';

describe('ProjectsPage', () => {
  let component: ProjectsPage;
  let fixture: ComponentFixture<ProjectsPage>;
  let projectServiceSpy: jasmine.SpyObj<ProjectService>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let routerSpy: jasmine.SpyObj<Router>;

  beforeEach(async () => {
    projectServiceSpy = jasmine.createSpyObj('ProjectService', ['getAll', 'delete']);
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getSummaryStats']);
    userServiceSpy = jasmine.createSpyObj('UserService', ['getIsAdmin', 'setActiveProject']);
    routerSpy = jasmine.createSpyObj('Router', ['navigate']);

    projectServiceSpy.getAll.and.returnValue(of([]));
    agentServiceSpy.getSummaryStats.and.returnValue(of({ total_cost: 0, total_sessions: 0 }));
    userServiceSpy.getIsAdmin.and.returnValue(true);

    await TestBed.configureTestingModule({
      imports: [ProjectsPage, CommonModule],
      providers: [
        { provide: ProjectService, useValue: projectServiceSpy },
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        { provide: Router, useValue: routerSpy }
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

  it('should navigate to edit page', () => {
    component.selectedProject = { id: '1' } as Project;
    component.editProject();
    expect(routerSpy.navigate).toHaveBeenCalledWith(['/project-edit', '1']);
  });

  it('should navigate to create page', () => {
    component.startNewProject();
    expect(routerSpy.navigate).toHaveBeenCalledWith(['/project-edit', 'new']);
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
});
