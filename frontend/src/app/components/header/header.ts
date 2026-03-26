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
import { Component, inject, OnInit } from '@angular/core';
import { Router, RouterLink, ActivatedRoute, NavigationEnd } from '@angular/router';
import { UserInfo } from '../user-info/user-info';
import { HeaderDropdown } from './header-dropdown/header-dropdown';
import { NavigationService } from '../../services/navigation-service';
import { ProjectService } from '../../services/project-service';
import { UserService } from '../../services/user-service';
import { Project } from '../../models/project';
import { CommonModule } from '@angular/common';
import { filter, map, mergeMap } from 'rxjs/operators';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [UserInfo, HeaderDropdown, CommonModule, RouterLink],
  templateUrl: './header.html',
  styleUrl: './header.css',
})
export class Header implements OnInit {
  private navigationService = inject(NavigationService);
  private projectService = inject(ProjectService);
  private userService = inject(UserService);
  private router = inject(Router);
  private activatedRoute = inject(ActivatedRoute);

  breadcrumbs = this.navigationService.breadcrumbs;
  projects: Project[] = [];
  activeProjectId: string | null = null;
  isProjectRoute = true;

  ngOnInit(): void {
    // Listen for route changes to update title from route data
    this.router.events.pipe(
      filter(event => event instanceof NavigationEnd),
      map(() => this.activatedRoute),
      map(route => {
        while (route.firstChild) route = route.firstChild;
        return route;
      }),
      mergeMap(route => route.data)
    ).subscribe(data => {
      if (data && data['title']) {
        this.navigationService.setTitle(data['title']);
      }
      this.isProjectRoute = data && data['isProjectRoute'] !== false;
    });

    this.userService.userDetails.subscribe(details => {
      if (details) {
        // Set active project first so template can reflect it while projects load
        this.activeProjectId = details.active_project_id && details.active_project_id !== 'None' ? details.active_project_id : null;
        this.projectService.getAll().subscribe((data: Project[]) => {
          this.projects = data;
          const activeProjectId = this.userService.getActiveProjectId();
          if (activeProjectId) {
            this.activeProjectId = activeProjectId;
          }
        });
      } else {
        this.projects = [];
        this.activeProjectId = null;
      }
    });
  }

  get activeProjectName(): string | null {
    if (!this.activeProjectId) return null;
    return this.projects.find(p => p.id === this.activeProjectId)?.name || null;
  }
}
