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
import { of, throwError } from 'rxjs';
import { Agent } from '../../models/agent';

describe('AgentsPage', () => {
  let component: AgentsPage;
  let fixture: ComponentFixture<AgentsPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let routerSpy: jasmine.SpyObj<Router>;

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getAgentsByProject', 'getAgentSvgUrl', 'getAgentStats']);
    userServiceSpy = jasmine.createSpyObj('UserService', ['getActiveProjectId']);
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
    userServiceSpy.getActiveProjectId.and.returnValue(null);
    fixture.detectChanges();
    expect(component.error).toBe('No active project selected');
  });

  it('should load agents if active project exists', () => {
    const mockAgents: Agent[] = [{ id: '1', name: 'Agent 1' } as Agent];
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));

    fixture.detectChanges();

    expect(agentServiceSpy.getAgentsByProject).toHaveBeenCalledWith('proj1');
    expect(component.agents).toEqual(mockAgents);
    expect(component.loading).toBeFalse();
  });

  it('should handle error when loading agents', () => {
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    agentServiceSpy.getAgentsByProject.and.returnValue(throwError(() => new Error('API Error')));

    // Suppress console error in test output
    spyOn(console, 'error');

    fixture.detectChanges();

    expect(component.error).toBe('Failed to load agents');
    expect(component.loading).toBeFalse();
  });

  it('should prepare chart data with DD-MM labels', () => {
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    const mockAgents: Agent[] = [{ id: '1', name: 'Agent 1' } as Agent];
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    const mockStats = {
      runs: [{ date: '2023-01-25', count: 5 }],
      sessions: [{ date: '2023-01-25', count: 2 }],
      messages: [{ date: '2023-01-25', count: 10 }]
    };
    agentServiceSpy.getAgentStats.and.returnValue(of(mockStats));

    fixture.detectChanges();

    expect(component.lineChartData.labels).toContain('25-01');
  });

  it('should navigate to conversations with agentId', () => {
    component.selectedAgent = { id: 'agent-123', name: 'Agent 123' } as Agent;
    component.goToConversations();
    expect(routerSpy.navigate).toHaveBeenCalledWith(['/conversations'], { queryParams: { agentId: 'agent-123' } });
  });
});
