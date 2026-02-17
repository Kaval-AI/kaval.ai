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
import { Routes } from '@angular/router';
import { ConversationsPage } from './components/conversations-page/conversations-page';
import { TestsPage } from './components/tests-page/tests-page';
import { MetricsPage } from './components/metrics-page/metrics-page';
import { LlmCallStatsPage } from './components/llm-call-stats-page/llm-call-stats-page';
import { RagPage } from './components/rag-page/rag-page';
import { SessionDetailPage } from './components/session-detail-page/session-detail-page';

import { ProjectsPage } from './components/projects-page/projects-page';
import { AgentsPage } from './components/agents-page/agents-page';
import { ProjectEditPage } from './components/project-edit-page/project-edit-page';
import { UserEditPage } from './components/user-edit-page/user-edit-page';
import { UsersPage } from './components/users-page/users-page';

export const routes: Routes = [
  // Default route
  { path: '', component: ProjectsPage },

  { path: 'agents', component: AgentsPage },
  { path: 'users', component: UsersPage },
  { path: 'project-edit/:id', component: ProjectEditPage },
  { path: 'user-edit/:id', component: UserEditPage },
  { path: 'conversations', component: ConversationsPage },
  { path: 'conversations/:sessionId', component: SessionDetailPage },
  { path: 'tests', component: TestsPage },
  { path: 'metrics', component: MetricsPage },
  { path: 'llm-call-stats', component: LlmCallStatsPage },
  { path: 'rag', component: RagPage },

  // Wildcard route for 404 - Should always be last
  { path: '**', redirectTo: '' },
];
