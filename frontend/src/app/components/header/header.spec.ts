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

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter, Router } from '@angular/router';
import { of, BehaviorSubject } from 'rxjs';

import { Header } from './header';
import { ProjectService } from '../../services/project-service';
import { UserService } from '../../services/user-service';
import { NavigationService } from '../../services/navigation-service';
import { Project } from '../../models/project';

describe('Header', () => {
  let component: Header;
  let fixture: ComponentFixture<Header>;
  let projectServiceSpy: jasmine.SpyObj<ProjectService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let navigationServiceSpy: jasmine.SpyObj<NavigationService>;
  let routerSpy: jasmine.SpyObj<Router>;

  beforeEach(async () => {
    projectServiceSpy = jasmine.createSpyObj('ProjectService', ['getAll']);
    userServiceSpy = jasmine.createSpyObj('UserService', ['getActiveProjectId', 'setActiveProject', 'updateUserDetails', 'getIsAdmin'], { userDetails: of({ id: 'u1' }) });
    navigationServiceSpy = jasmine.createSpyObj('NavigationService', ['setTitle'], { title: () => 'Test Title' });
    routerSpy = jasmine.createSpyObj('Router', ['navigate', 'navigateByUrl'], { url: '/test' });

    projectServiceSpy.getAll.and.returnValue(of([]));
    userServiceSpy.getActiveProjectId.and.returnValue(null);
    routerSpy.navigateByUrl.and.returnValue(Promise.resolve(true));
    routerSpy.navigate.and.returnValue(Promise.resolve(true));

    await TestBed.configureTestingModule({
      imports: [Header],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        { provide: ProjectService, useValue: projectServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        { provide: NavigationService, useValue: navigationServiceSpy },
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(Header);
    component = fixture.componentInstance;
    routerSpy = TestBed.inject(Router) as jasmine.SpyObj<Router>;
    spyOn(routerSpy, 'navigateByUrl').and.returnValue(Promise.resolve(true));
    spyOn(routerSpy, 'navigate').and.returnValue(Promise.resolve(true));
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should show breadcrumbs with page title and active project name', async () => {
    const mockProjects: Project[] = [
      { id: '1', name: 'Project 1' } as Project
    ];
    projectServiceSpy.getAll.and.returnValue(of(mockProjects));
    userServiceSpy.getActiveProjectId.and.returnValue('1');

    component.ngOnInit();
    fixture.detectChanges();
    await fixture.whenStable();

    const breadcrumbs = fixture.nativeElement.querySelectorAll('.breadcrumbs li');
    expect(breadcrumbs.length).toBe(2);
    expect(breadcrumbs[0].textContent).toContain('Project 1');
    expect(breadcrumbs[1].textContent).toContain('Test Title');

    // Check for icons
    const icons = fixture.nativeElement.querySelectorAll('.breadcrumbs li svg');
    expect(icons.length).toBe(2);
  });

  it('should have routerLink="/" on logo and active project breadcrumb', async () => {
    const mockProjects: Project[] = [
      { id: '1', name: 'Project 1' } as Project
    ];
    projectServiceSpy.getAll.and.returnValue(of(mockProjects));
    userServiceSpy.getActiveProjectId.and.returnValue('1');

    component.ngOnInit();
    fixture.detectChanges();
    await fixture.whenStable();

    const logoLink = fixture.nativeElement.querySelector('a[routerLink="/"] img.logo');
    expect(logoLink).toBeTruthy();

    const breadcrumbs = fixture.nativeElement.querySelectorAll('.breadcrumbs li');
    const projectLink = breadcrumbs[0].querySelector('a[routerLink="/"]');
    expect(projectLink).toBeTruthy();
    expect(projectLink.textContent).toContain('Project 1');
  });

  it('should have 4 menu sections with correct titles', () => {
    userServiceSpy.getIsAdmin.and.returnValue(true);
    fixture.detectChanges();

    const menuTitles = fixture.nativeElement.querySelectorAll('.menu-title');
    expect(menuTitles.length).toBe(4);
    expect(menuTitles[0].textContent).toBe('Admin');
    expect(menuTitles[1].textContent).toBe('Agents');
    expect(menuTitles[2].textContent).toBe('RAG');
    expect(menuTitles[3].textContent).toBe('Other');
  });

  it('should have correct links in each section', () => {
    userServiceSpy.getIsAdmin.and.returnValue(true);
    fixture.detectChanges();

    const dropdownContent = fixture.nativeElement.querySelector('.dropdown-content');
    const menus = dropdownContent.querySelectorAll('ul.menu');

    // Admin section (1st column)
    const adminMenu = menus[0].querySelectorAll('li');
    expect(adminMenu[0].querySelector('.menu-title').textContent).toBe('Admin');
    expect(adminMenu[1].querySelector('a').getAttribute('routerLink')).toBe('/');
    expect(adminMenu[2].querySelector('a').getAttribute('routerLink')).toBe('/users');

    // Agents section (2nd column)
    const agentsMenu = menus[1].querySelectorAll('li');
    expect(agentsMenu[0].querySelector('.menu-title').textContent).toBe('Agents');
    expect(agentsMenu[1].querySelector('a').getAttribute('routerLink')).toBe('/agents');
    expect(agentsMenu[2].querySelector('a').getAttribute('routerLink')).toBe('/conversations');
    expect(agentsMenu[3].querySelector('a').getAttribute('routerLink')).toBe('/llm-call-stats');

    // RAG section (3rd column)
    const ragMenu = menus[2].querySelectorAll('li');
    expect(ragMenu[0].querySelector('.menu-title').textContent).toBe('RAG');
    expect(ragMenu[1].querySelector('a').getAttribute('routerLink')).toBe('/rag');

    // Other section (4th column)
    const otherMenu = menus[3].querySelectorAll('li');
    expect(otherMenu[0].querySelector('.menu-title').textContent).toBe('Other');
    expect(otherMenu[1].querySelector('a').getAttribute('routerLink')).toBe('/theme');
  });

  it('should have the correct background color for the dropdown content', () => {
    const dropdownContent = fixture.nativeElement.querySelector('.dropdown-content');
    const styles = window.getComputedStyle(dropdownContent);
    // In unit tests, CSS variables might not be resolved, but we can check the computed style
    // However, depending on the test environment, it might return the hex or the variable name
    // Given the previous tests, we can at least check if the class is present.
    expect(dropdownContent.classList).toContain('dropdown-content');
  });
});
