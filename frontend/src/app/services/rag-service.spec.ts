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

import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { RagService } from './rag-service';
import { RagResult, RagStats } from '../models/rag';

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

  it('should query RAG', () => {
    const mockResults: RagResult[] = [
      { content: 'Result 1' } as RagResult
    ];
    const projectId = 'proj123';
    const queryData = { model: 'text-embedding-3-small', text: 'query' };

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
