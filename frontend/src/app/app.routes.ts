import { Routes } from '@angular/router';
import { ConversationsPage } from './components/conversations-page/conversations-page';
import { TestsPage } from './components/tests-page/tests-page';
import { MetricsPage } from './components/metrics-page/metrics-page';
import { ConfigsPage } from './components/configs-page/configs-page';
import { RagPage } from './components/rag-page/rag-page';
import { SessionDetailPage } from './components/session-detail-page/session-detail-page';

import { LandingPage } from './components/landing-page/landing-page';
import { AgentsPage } from './components/agents-page/agents-page';
import { ProjectEditPage } from './components/project-edit-page/project-edit-page';

export const routes: Routes = [
  // Default route
  { path: '', component: LandingPage },

  { path: 'agents', component: AgentsPage },
  { path: 'project-edit/:id', component: ProjectEditPage },
  { path: 'conversations', component: ConversationsPage },
  { path: 'conversations/:sessionId', component: SessionDetailPage },
  { path: 'tests', component: TestsPage },
  { path: 'metrics', component: MetricsPage },
  { path: 'configs', component: ConfigsPage },
  { path: 'rag', component: RagPage },

  // Wildcard route for 404 - Should always be last
  { path: '**', redirectTo: '' },
];
