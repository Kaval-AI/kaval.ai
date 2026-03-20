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
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RagService } from '../../services/rag-service';
import { UserService } from '../../services/user-service';
import { RagResult, RagStats } from '../../models/rag';

@Component({
  selector: 'app-rag-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './rag-page.html',
  styleUrl: './rag-page.css',
})
export class RagPage implements OnInit {
  projectId: string | null = null;
  results: RagResult[] = [];
  ragStats: RagStats | null = null;

  queryText: string = '';
  selectedModel: string = 'openai/text-embedding-3-small';
  collectionName: string = '';
  sourceIdsInput: string = '';
  topK: number = 10;
  keepBest: boolean = false;
  normalizerYaml: string = '';

  loading: boolean = false;
  error: string | null = null;

  constructor(
    private ragService: RagService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.userService.userDetails.subscribe(user => {
      if (user && user.active_project_id) {
        const newProjectId = user.active_project_id !== 'None' ? user.active_project_id : null;
        if (newProjectId !== this.projectId) {
          this.projectId = newProjectId;
          this.results = [];
          this.loadRagStats();
        }
      }
    });
  }

  loadRagStats(): void {
    if (!this.projectId) return;

    this.ragService.getRagStats(this.projectId).subscribe({
      next: (stats) => {
        this.ragStats = stats;
      },
      error: (err) => {
        console.error('Error loading RAG stats', err);
      }
    });
  }

  onQuery(): void {
    if (!this.projectId || !this.selectedModel || !this.queryText) {
      return;
    }

    this.loading = true;
    this.error = null;
    const sourceIds = this.sourceIdsInput
      ? this.sourceIdsInput.split(',').map(id => id.trim()).filter(id => !!id)
      : undefined;

    this.ragService.queryRag(this.projectId, {
      model: this.selectedModel,
      text: this.queryText,
      collection_name: this.collectionName || undefined,
      top_k: this.topK,
      source_ids: sourceIds,
      keep_best: this.keepBest,
      normalizer_yaml: this.normalizerYaml || undefined
    }).subscribe({
      next: (results) => {
        this.results = results;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error querying RAG', err);
        this.error = 'Failed to execute RAG query.';
        this.loading = false;
      }
    });
  }
}
