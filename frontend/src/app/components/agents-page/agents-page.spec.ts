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
import { AgentsPage } from './agents-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Router } from '@angular/router';
import { of, throwError, BehaviorSubject } from 'rxjs';
import { Agent } from '../../models/agent';

describe('AgentsPage', () => {
  let component: AgentsPage;
  let fixture: ComponentFixture<AgentsPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: any;
  let routerSpy: jasmine.SpyObj<Router>;

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getAgentsByProject', 'getSummaryStats', 'getDailyStats']);
    userServiceSpy = {
      userDetails: new BehaviorSubject<any>({ active_project_id: null }),
      getActiveProjectId: jasmine.createSpy('getActiveProjectId')
    };
    routerSpy = jasmine.createSpyObj('Router', ['navigate']);

    await TestBed.configureTestingModule({
      imports: [AgentsPage],
      providers: [
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        { provide: Router, useValue: routerSpy }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(AgentsPage);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    userServiceSpy.getActiveProjectId.and.returnValue(null);
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should show error if no active project', () => {
    userServiceSpy.userDetails.next({ active_project_id: 'None' });
    fixture.detectChanges();
    expect(component.error).toBe('No active project selected');
  });

  it('should load agents if active project exists', () => {
    const mockAgents: Agent[] = [{ id: '1', name: 'Agent 1' } as Agent];
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSummaryStats.and.returnValue(of({}));
    agentServiceSpy.getDailyStats.and.returnValue(of({
      sessions: [], messages: [], tasks: [], runs: {}, llm: {}, embedding: {}
    }));

    userServiceSpy.userDetails.next({ active_project_id: 'proj1' });
    fixture.detectChanges();

    expect(agentServiceSpy.getAgentsByProject).toHaveBeenCalledWith('proj1');
    expect(component.agents).toEqual(mockAgents);
    expect(component.loading).toBeFalse();
  });

  it('should handle error when loading agents', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(throwError(() => new Error('API Error')));

    // Suppress console error in test output
    spyOn(console, 'error');

    userServiceSpy.userDetails.next({ active_project_id: 'proj1' });
    fixture.detectChanges();

    expect(component.error).toBe('Failed to load agents');
    expect(component.loading).toBeFalse();
  });

  it('should prepare chart data with DD-MM labels', () => {
    const mockAgents: Agent[] = [{ id: '1', name: 'Agent 1' } as Agent];
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSummaryStats.and.returnValue(of({}));
    const mockDailyStats = {
      sessions: [{ date: '2023-01-25', count: 2 }],
      messages: [{ date: '2023-01-25', count: 10 }],
      tasks: [{ date: '2023-01-25', count: 5 }],
      runs: { 'Agent 1': [{ date: '2023-01-25', count: 5 }] },
      llm: {},
      embedding: {}
    };
    agentServiceSpy.getDailyStats.and.returnValue(of(mockDailyStats));

    userServiceSpy.userDetails.next({ active_project_id: 'proj1' });
    fixture.detectChanges();

    expect(component.activityChartData.labels).toContain('25-01');
  });

  it('should navigate to conversations with agentId', () => {
    component.selectedAgent = { id: 'agent-123', name: 'Agent 123' } as Agent;
    component.goToConversations();
    expect(routerSpy.navigate).toHaveBeenCalledWith(['/conversations'], { queryParams: { agentId: 'agent-123' } });
  });

  it('should open modal with correct data', () => {
    const mockAgent = {
      id: '1',
      name: 'Agent 1',
      workflow: { steps: [] },
      input_schema: { type: 'object' },
      output_schema: { type: 'string' }
    } as Agent;
    component.selectedAgent = mockAgent;

    component.openModal('workflow');
    expect(component.showModal).toBeTrue();
    expect(component.modalTitle).toBe('Workflow JSON');
    expect(component.modalData).toEqual(mockAgent.workflow);

    component.openModal('input');
    expect(component.modalTitle).toBe('Input Schema');
    expect(component.modalData).toEqual(mockAgent.input_schema);

    component.openModal('output');
    expect(component.modalTitle).toBe('Output Schema');
    expect(component.modalData).toEqual(mockAgent.output_schema);
  });

  it('should close modal', () => {
    component.showModal = true;
    component.modalData = { some: 'data' };
    component.closeModal();
    expect(component.showModal).toBeFalse();
    expect(component.modalData).toBeNull();
  });
});
