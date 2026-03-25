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

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
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
    projectServiceSpy = jasmine.createSpyObj('ProjectService', ['getAll', 'delete', 'update', 'getMembers', 'addMember', 'updateMember', 'removeMember']);
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getSummaryStats', 'getDailyStats']);
    userServiceSpy = jasmine.createSpyObj('UserService', ['getIsAdmin', 'setActiveProject', 'getActiveProjectId', 'getUsers'], { userDetails: of({ id: 'u1' }) });
    routerSpy = jasmine.createSpyObj('Router', ['navigate']);

    projectServiceSpy.getAll.and.returnValue(of([]));
    projectServiceSpy.getMembers.and.returnValue(of([]));
    agentServiceSpy.getSummaryStats.and.returnValue(of({ total_cost: null, llm_cost: null, embedding_cost: null, total_sessions: 0 }));
    agentServiceSpy.getDailyStats.and.returnValue(of({
      sessions: [],
      messages: [],
      tasks: [],
      runs: {},
      llm: {},
      embedding: {}
    }));
    userServiceSpy.getIsAdmin.and.returnValue(true);
    userServiceSpy.getActiveProjectId.and.returnValue(null);
    userServiceSpy.getUsers.and.returnValue(of([]));

    await TestBed.configureTestingModule({
      imports: [ProjectsPage, CommonModule, FormsModule],
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

  it('should load projects on init and select first', () => {
    const mockProjects: Project[] = [{ id: '1', name: 'Project 1' } as Project];
    projectServiceSpy.getAll.and.returnValue(of(mockProjects));
    userServiceSpy.getActiveProjectId.and.returnValue(null);

    component.ngOnInit();

    expect(projectServiceSpy.getAll).toHaveBeenCalled();
    expect(component.projects).toEqual(mockProjects);
    expect(component.selectedProject).toEqual(mockProjects[0]);
  });

  it('should select active project on init if it exists', () => {
    const mockProjects: Project[] = [
      { id: '1', name: 'Project 1' } as Project,
      { id: '2', name: 'Project 2' } as Project
    ];
    projectServiceSpy.getAll.and.returnValue(of(mockProjects));
    userServiceSpy.getActiveProjectId.and.returnValue('2');

    component.ngOnInit();

    expect(component.selectedProject).toEqual(mockProjects[1]);
  });

  it('should enter edit mode and open access details', () => {
    component.selectedProject = { id: '1', name: 'P1' } as Project;
    component.accessDetailsCollapsed = true;
    component.editProject();
    expect(component.isEditing).toBeTrue();
    expect(component.editableProject.name).toBe('P1');
    expect(component.accessDetailsCollapsed).toBeFalse();
  });

  it('should save project and exit edit mode', () => {
    const originalProject = { id: '1', name: 'P1' } as Project;
    const updatedProject = { id: '1', name: 'Updated' } as Project;
    component.selectedProject = originalProject;
    component.editableProject = { name: 'Updated' };
    component.isEditing = true;
    projectServiceSpy.update.and.returnValue(of(updatedProject));
    projectServiceSpy.getAll.and.returnValue(of([updatedProject]));

    component.saveProject();

    expect(projectServiceSpy.update).toHaveBeenCalledWith('1', { name: 'Updated' });
    expect(component.isEditing).toBeFalse();
    expect(component.selectedProject?.name).toBe('Updated');
  });

  it('should navigate to create page', () => {
    component.startNewProject();
    expect(routerSpy.navigate).toHaveBeenCalledWith(['/project-edit', 'new']);
  });

  it('should delete a project after confirmation', () => {
    const existingProject = { id: '1', name: 'Existing' } as Project;
    component.selectedProject = existingProject;
    component.deleteConfirmationName = 'Existing';
    projectServiceSpy.delete.and.returnValue(of(undefined));
    projectServiceSpy.getAll.and.returnValue(of([]));

    component.deleteProject();

    expect(projectServiceSpy.delete).toHaveBeenCalledWith('1');
    expect(component.selectedProject).toBeNull();
  });

  it('should not delete a project if name does not match', () => {
    const existingProject = { id: '1', name: 'Existing' } as Project;
    component.selectedProject = existingProject;
    component.deleteConfirmationName = 'Wrong';

    component.deleteProject();

    expect(projectServiceSpy.delete).not.toHaveBeenCalled();
  });

  it('should handle project selection from event', () => {
    const mockProjects: Project[] = [
      { id: '1', name: 'P1' } as Project,
      { id: '2', name: 'P2' } as Project
    ];
    component.projects = mockProjects;

    component.selectProject(mockProjects[1]);

    expect(component.selectedProject?.id).toBe('2');
    expect(projectServiceSpy.getMembers).toHaveBeenCalledWith('2');
  });

  it('should handle project selection from select element', () => {
    const mockProjects: Project[] = [
      { id: '1', name: 'P1' } as Project,
      { id: '2', name: 'P2' } as Project
    ];
    component.projects = mockProjects;
    const event = { target: { value: '2' } } as any;

    component.onProjectSelect(event);

    expect(userServiceSpy.setActiveProject).toHaveBeenCalledWith('2');
    expect(component.selectedProject?.id).toBe('2');
  });

  it('should toggle access details section collapse', () => {
    expect(component.accessDetailsCollapsed).toBeTrue();
    component.accessDetailsCollapsed = false;
    expect(component.accessDetailsCollapsed).toBeFalse();
    component.accessDetailsCollapsed = true;
    expect(component.accessDetailsCollapsed).toBeTrue();
  });

  it('should toggle connection strings section collapse', () => {
    expect(component.connectionStringsCollapsed).toBeTrue();
    component.connectionStringsCollapsed = false;
    expect(component.connectionStringsCollapsed).toBeFalse();
    component.connectionStringsCollapsed = true;
    expect(component.connectionStringsCollapsed).toBeTrue();
  });
});
