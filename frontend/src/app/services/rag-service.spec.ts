import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { RagService } from './rag-service';
import { EmbeddingConfig, RagResult, RagStats } from '../models/rag';

describe('RagService', () => {
  let service: RagService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [RagService]
    });
    service = TestBed.inject(RagService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should fetch embedding configs', () => {
    const mockConfigs: EmbeddingConfig[] = [
      { id: '1', name: 'Config 1' } as EmbeddingConfig
    ];
    const projectId = 'proj123';

    service.getEmbeddingConfigs(projectId).subscribe(configs => {
      expect(configs).toEqual(mockConfigs);
    });

    const req = httpMock.expectOne(`/api/projects/${projectId}/embedding-configs`);
    expect(req.request.method).toBe('GET');
    req.flush(mockConfigs);
  });

  it('should query RAG', () => {
    const mockResults: RagResult[] = [
      { content: 'Result 1' } as RagResult
    ];
    const projectId = 'proj123';
    const queryData = { embedding_profile_id: 'prof1', text: 'query' };

    service.queryRag(projectId, queryData).subscribe(results => {
      expect(results).toEqual(mockResults);
    });

    const req = httpMock.expectOne(`/api/projects/${projectId}/rag/query`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(queryData);
    req.flush(mockResults);
  });

  it('should fetch RAG stats', () => {
    const mockStats: RagStats = { total_entries: 10, total_collections: 1, collections: ['default'] };
    const projectId = 'proj123';

    service.getRagStats(projectId).subscribe(stats => {
      expect(stats).toEqual(mockStats);
    });

    const req = httpMock.expectOne(`/api/projects/${projectId}/rag/stats`);
    expect(req.request.method).toBe('GET');
    req.flush(mockStats);
  });
});
