import { ComponentFixture, TestBed } from '@angular/core/testing';
import { AgentsPage } from './agents-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { of, throwError } from 'rxjs';
import { Agent } from '../../models/agent';

describe('AgentsPage', () => {
  let component: AgentsPage;
  let fixture: ComponentFixture<AgentsPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getAgentsByProject', 'getAgentSvgUrl', 'getAgentStats']);
    userServiceSpy = jasmine.createSpyObj('UserService', ['getActiveProjectId']);

    await TestBed.configureTestingModule({
      imports: [AgentsPage],
      providers: [
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: userServiceSpy }
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
});
