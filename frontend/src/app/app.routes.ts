import { Routes } from '@angular/router';
import { ProjectsPage } from './components/projects-page/projects-page';
import { UsersPage } from './components/users-page/users-page';
import { AgentsPage } from './components/agents-page/agents-page';
import { ConversationsPage } from './components/conversations-page/conversations-page';
import { TestsPage } from './components/tests-page/tests-page';
import { MetricsPage } from './components/metrics-page/metrics-page';
import { ConfigsPage } from './components/configs-page/configs-page';
import { RagPage } from './components/rag-page/rag-page';

export const routes: Routes = [
  // Default route (redirects empty path to projects)
  { path: '', redirectTo: 'projects', pathMatch: 'full' },

  { path: 'projects', component: ProjectsPage },
  { path: 'users', component: UsersPage },
  { path: 'agents', component: AgentsPage },
  { path: 'conversations', component: ConversationsPage },
  { path: 'tests', component: TestsPage },
  { path: 'metrics', component: MetricsPage },
  { path: 'configs', component: ConfigsPage },
  { path: 'rag', component: RagPage },

  // Wildcard route for 404 - Should always be last
  { path: '**', redirectTo: 'projects' }
];
