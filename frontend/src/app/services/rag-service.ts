import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { EmbeddingConfig, RagResult, RagStats } from '../models/rag';

@Injectable({
  providedIn: 'root'
})
export class RagService {
  constructor(private http: HttpClient) {}

  getEmbeddingConfigs(projectId: string): Observable<EmbeddingConfig[]> {
    return this.http.get<EmbeddingConfig[]>(`/api/projects/${projectId}/embedding-configs`);
  }

  queryRag(projectId: string, queryData: {
    embedding_profile_id: string,
    text: string,
    collection_name?: string,
    top_k?: number
  }): Observable<RagResult[]> {
    return this.http.post<RagResult[]>(`/api/projects/${projectId}/rag/query`, queryData);
  }

  getRagStats(projectId: string): Observable<RagStats> {
    return this.http.get<RagStats>(`/api/projects/${projectId}/rag/stats`);
  }
}
