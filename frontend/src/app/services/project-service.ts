import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Project } from '../models/project';

@Injectable({
  providedIn: 'root',
})
export class ProjectService {
  private http = inject(HttpClient);

  create(project: Partial<Project>): Observable<Project> {
    // We send a partial object, but receive a full Project (with ID) back
    return this.http.post<Project>("/api/projects/create", project);
  }

  getById(id: string | number): Observable<Project> {
    return this.http.get<Project>(`/api/projects/get/${id}`);
  }

  getAll(): Observable<Project[]> {
    return this.http.get<Project[]>("/api/projects/all");
  }

  update(id: string, project: Partial<Project>): Observable<Project> {
    return this.http.put<Project>(`/api/projects/update/${id}`, project);
  }

  delete(id: string | number): Observable<void> {
    return this.http.delete<void>(`/api/projects/delete/${id}`);
  }

  testConnection(projectId: string): Observable<{status: string, message?: string}> {
    return this.http.post<{status: string, message?: string}>(`/api/projects/test-connection/${projectId}`, {});
  }

  getMembers(projectId: string): Observable<any[]> {
    return this.http.get<any[]>(`/api/projects/${projectId}/members`);
  }

  addMember(projectId: string, userId: string, role: string): Observable<any> {
    return this.http.post<any>(`/api/projects/${projectId}/members/add`, { user_id: userId, role });
  }

  updateMember(projectId: string, userId: string, role: string): Observable<any> {
    return this.http.put<any>(`/api/projects/${projectId}/members/update`, { user_id: userId, role });
  }

  removeMember(projectId: string, userId: string): Observable<any> {
    return this.http.delete<any>(`/api/projects/${projectId}/members/remove/${userId}`);
  }
}
