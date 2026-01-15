import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { AgentService } from './agent-service';
import { Agent } from '../models/agent';

describe('AgentService', () => {
  let service: AgentService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [AgentService]
    });
    service = TestBed.inject(AgentService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should fetch agents by project id', () => {
    const mockAgents: Agent[] = [
      { id: '1', name: 'Agent 1' } as Agent,
      { id: '2', name: 'Agent 2' } as Agent
    ];
    const projectId = 'proj123';

    service.getAgentsByProject(projectId).subscribe(agents => {
      expect(agents.length).toBe(2);
      expect(agents).toEqual(mockAgents);
    });

    const req = httpMock.expectOne(`/api/agents/all/${projectId}`);
    expect(req.request.method).toBe('GET');
    req.flush(mockAgents);
  });
});
