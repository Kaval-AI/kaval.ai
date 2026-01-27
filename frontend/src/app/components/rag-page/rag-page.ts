import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RagService } from '../../services/rag-service';
import { UserService } from '../../services/user-service';
import { EmbeddingConfig, RagResult, RagStats } from '../../models/rag';

@Component({
  selector: 'app-rag-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './rag-page.html',
  styleUrl: './rag-page.css',
})
export class RagPage implements OnInit {
  projectId: string | null = null;
  embeddingConfigs: EmbeddingConfig[] = [];
  results: RagResult[] = [];
  ragStats: RagStats | null = null;

  queryText: string = '';
  selectedEmbeddingProfileId: string = '';
  collectionName: string = '';
  topK: number = 10;

  loading: boolean = false;
  error: string | null = null;

  constructor(
    private ragService: RagService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.projectId = this.userService.getActiveProjectId();
    if (this.projectId) {
      this.loadEmbeddingConfigs();
      this.loadRagStats();
    }
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

  loadEmbeddingConfigs(): void {
    if (!this.projectId) return;

    this.ragService.getEmbeddingConfigs(this.projectId).subscribe({
      next: (configs) => {
        this.embeddingConfigs = configs;
        if (configs.length > 0) {
          this.selectedEmbeddingProfileId = configs[0].id;
        }
      },
      error: (err) => {
        console.error('Error loading embedding configs', err);
        this.error = 'Failed to load embedding profiles.';
      }
    });
  }

  onQuery(): void {
    if (!this.projectId || !this.selectedEmbeddingProfileId || !this.queryText) {
      return;
    }

    this.loading = true;
    this.error = null;
    this.ragService.queryRag(this.projectId, {
      embedding_profile_id: this.selectedEmbeddingProfileId,
      text: this.queryText,
      collection_name: this.collectionName || undefined,
      top_k: this.topK
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
