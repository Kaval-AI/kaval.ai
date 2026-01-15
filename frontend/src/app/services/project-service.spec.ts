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
});
