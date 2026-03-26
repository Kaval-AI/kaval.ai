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
import { RunTasksPage } from './run-tasks-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { ActivatedRoute, Router, convertToParamMap, provideRouter } from '@angular/router';
import { of, BehaviorSubject, throwError } from 'rxjs';
import { Task } from '../../models/task';
import { TasksList } from '../tasks-list/tasks-list';

describe('RunTasksPage', () => {
  let component: RunTasksPage;
  let fixture: ComponentFixture<RunTasksPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let router: Router;
  let userDetailsSubject: BehaviorSubject<any>;

  const mockTasks: Task[] = [
    { id: 'task1', agent_id: 'agent1', session_id: 'sess1', run_id: 'run1', name: 'Task 1', created_at: new Date().toISOString(), updated_at: new Date().toISOString() } as Task,
    { id: 'task2', agent_id: 'agent1', session_id: 'sess1', run_id: 'run1', name: 'Task 2', created_at: new Date().toISOString(), updated_at: new Date().toISOString() } as Task,
    { id: 'task3', agent_id: 'agent1', session_id: 'sess1', run_id: 'run2', name: 'Task 3', created_at: new Date().toISOString(), updated_at: new Date().toISOString() } as Task
  ];

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getSessionDetails']);
    userDetailsSubject = new BehaviorSubject({ active_project_id: 'proj1' });
    userServiceSpy = {
      userDetails: userDetailsSubject.asObservable()
    } as any;

    await TestBed.configureTestingModule({
      imports: [RunTasksPage, TasksList],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        {
          provide: ActivatedRoute,
          useValue: {
            paramMap: of(convertToParamMap({ sessionId: 'sess1', runId: 'run1' }))
          }
        }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(RunTasksPage);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    spyOn(router, 'navigate');
  });

  it('should create', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of({ session_id: 'sess1', tasks: [], messages: [], runs: [] }));
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should load tasks for the correct run on init', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of({ session_id: 'sess1', tasks: mockTasks, messages: [], runs: [] }));
    fixture.detectChanges();

    expect(agentServiceSpy.getSessionDetails).toHaveBeenCalledWith('proj1', 'sess1');
    expect(component.tasks.length).toBe(2);
    expect(component.tasks.every(t => t.run_id === 'run1')).toBeTrue();
  });

  it('should handle error when loading tasks', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(throwError(() => new Error('API Error')));
    spyOn(console, 'error');
    fixture.detectChanges();

    expect(component.error).toBe('Failed to load tasks');
    expect(component.loading).toBeFalse();
  });

  it('should navigate back to conversation details when goBack is called', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of({ session_id: 'sess1', tasks: [], messages: [], runs: [] }));
    fixture.detectChanges();
    component.sessionId = 'sess1';
    component.goBack();
    expect(router.navigate).toHaveBeenCalledWith(['/conversations', 'sess1']);
  });

  it('should reload tasks when project changes', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of({ session_id: 'sess1', tasks: mockTasks, messages: [], runs: [] }));
    fixture.detectChanges();
    expect(agentServiceSpy.getSessionDetails).toHaveBeenCalledTimes(1);

    agentServiceSpy.getSessionDetails.calls.reset();
    userDetailsSubject.next({ active_project_id: 'proj2' });

    expect(agentServiceSpy.getSessionDetails).toHaveBeenCalledWith('proj2', 'sess1');
  });
});
