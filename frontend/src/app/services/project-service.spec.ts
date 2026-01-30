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
import { ProjectService } from './project-service';
import { Project } from '../models/project';

describe('ProjectService', () => {
  let service: ProjectService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [ProjectService]
    });
    service = TestBed.inject(ProjectService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should create a project', () => {
    const mockProject = { name: 'New Project' } as Project;
    service.create(mockProject).subscribe(project => {
      expect(project).toEqual(mockProject);
    });
    const req = httpMock.expectOne('/api/projects/create');
    expect(req.request.method).toBe('POST');
    req.flush(mockProject);
  });

  it('should get project by id', () => {
    const mockProject = { id: '1', name: 'Project 1' } as Project;
    service.getById('1').subscribe(project => {
      expect(project).toEqual(mockProject);
    });
    const req = httpMock.expectOne('/api/projects/get/1');
    expect(req.request.method).toBe('GET');
    req.flush(mockProject);
  });

  it('should get all projects', () => {
    const mockProjects = [{ id: '1', name: 'Project 1' }] as Project[];
    service.getAll().subscribe(projects => {
      expect(projects).toEqual(mockProjects);
    });
    const req = httpMock.expectOne('/api/projects/all');
    expect(req.request.method).toBe('GET');
    req.flush(mockProjects);
  });

  it('should update a project', () => {
    const mockProject = { id: '1', name: 'Updated' } as Project;
    service.update('1', mockProject).subscribe(project => {
      expect(project).toEqual(mockProject);
    });
    const req = httpMock.expectOne('/api/projects/update/1');
    expect(req.request.method).toBe('PUT');
    req.flush(mockProject);
  });

  it('should delete a project', () => {
    service.delete('1').subscribe();
    const req = httpMock.expectOne('/api/projects/delete/1');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });

  it('should test connection', () => {
    const mockRes = { status: 'success' };
    service.testConnection('1').subscribe(res => {
      expect(res).toEqual(mockRes);
    });
    const req = httpMock.expectOne('/api/projects/test-connection/1');
    expect(req.request.method).toBe('POST');
    req.flush(mockRes);
  });

  it('should handle getById with number id', () => {
    const mockProject = { id: '1', name: 'Project 1' } as Project;
    service.getById(1).subscribe(project => {
      expect(project).toEqual(mockProject);
    });
    const req = httpMock.expectOne('/api/projects/get/1');
    expect(req.request.method).toBe('GET');
    req.flush(mockProject);
  });

  it('should handle delete with number id', () => {
    service.delete(1).subscribe();
    const req = httpMock.expectOne('/api/projects/delete/1');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });
});
