import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ConversationsPage } from './conversations-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { of, throwError } from 'rxjs';
import { Agent } from '../../models/agent';
import { SessionSummary } from '../../models/session';

describe('ConversationsPage', () => {
  let component: ConversationsPage;
  let fixture: ComponentFixture<ConversationsPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;

  const mockAgents: Agent[] = [
    { id: 'agent1', name: 'Agent 1' } as Agent,
    { id: 'agent2', name: 'Agent 2' } as Agent
  ];

  const mockSessions: SessionSummary[] = [
    {
      session_id: 'sess1',
      agent_id: 'agent1',
      agent_name: 'Agent 1',
      runs_count: 1,
      tasks_count: 2,
      messages_count: 3,
      first_message: 'Hello',
      last_message: 'Bye',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }
  ];

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getAgentsByProject', 'getSessions']);
    userServiceSpy = jasmine.createSpyObj('UserService', ['getActiveProjectId']);

    await TestBed.configureTestingModule({
      imports: [ConversationsPage],
      providers: [
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: userServiceSpy }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(ConversationsPage);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    agentServiceSpy.getAgentsByProject.and.returnValue(of([]));
    agentServiceSpy.getSessions.and.returnValue(of([]));
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should load agents and sessions on init', () => {
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of(mockSessions));

    fixture.detectChanges();

    expect(userServiceSpy.getActiveProjectId).toHaveBeenCalled();
    expect(agentServiceSpy.getAgentsByProject).toHaveBeenCalledWith('proj1');
    expect(agentServiceSpy.getSessions).toHaveBeenCalledWith('proj1', undefined, 20, 0);
    expect(component.agents).toEqual(mockAgents);
    expect(component.sessions).toEqual(mockSessions);
  });

  it('should filter by agent when agent selection changes', () => {
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of(mockSessions));

    fixture.detectChanges();

    component.selectedAgentId = 'agent1';
    component.onAgentChange();

    expect(agentServiceSpy.getSessions).toHaveBeenCalledWith('proj1', 'agent1', 20, 0);
  });

  it('should set hasMore to true if exactly limit sessions are returned', () => {
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));

    // Create exactly 'limit' (20) sessions
    const twentySessions = Array(20).fill(mockSessions[0]);
    agentServiceSpy.getSessions.and.returnValue(of(twentySessions));

    fixture.detectChanges();

    expect(component.hasMore).toBeTrue();
  });

  it('should load more sessions when nextPage is called', () => {
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));

    // Initial load returns exactly 'limit' sessions so hasMore remains true
    const twentySessions = Array(20).fill(mockSessions[0]);
    agentServiceSpy.getSessions.and.returnValue(of(twentySessions));

    fixture.detectChanges();

    expect(component.hasMore).toBeTrue();

    // Reset spy to track second call
    agentServiceSpy.getSessions.calls.reset();
    agentServiceSpy.getSessions.and.returnValue(of([{ ...mockSessions[0], session_id: 'sess21' }]));

    component.nextPage();

    expect(component.offset).toBe(20);
    expect(agentServiceSpy.getSessions).toHaveBeenCalledWith('proj1', undefined, 20, 20);
    expect(component.sessions.length).toBe(21);
  });

  it('should handle error when loading sessions', () => {
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(throwError(() => new Error('API Error')));

    spyOn(console, 'error');
    fixture.detectChanges();

    expect(component.error).toBe('Failed to load sessions');
    expect(component.loading).toBeFalse();
  });

  it('should set hasMore to false if fewer than limit sessions are returned', () => {
    userServiceSpy.getActiveProjectId.and.returnValue('proj1');
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of(mockSessions)); // length 1 < limit 20

    fixture.detectChanges();

    expect(component.hasMore).toBeFalse();
  });

  it('should format date to DD-MM-YYYY HH:mm', () => {
    const dateStr = '2023-12-31T23:59:59Z';
    // Note: The result might depend on the timezone of the environment where tests run.
    // However, 2023-12-31T23:59:59Z should always result in 31/12/2023 or 01/01/2024 depending on TZ.
    // Let's use a more stable test that doesn't depend on TZ if possible, or just check the format.
    const result = component.formatDate(dateStr);
    expect(result).toMatch(/^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$/);
  });
});
