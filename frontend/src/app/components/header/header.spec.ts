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
    navigationServiceSpy = jasmine.createSpyObj('NavigationService', ['setTitle'], { breadcrumbs: () => [{ label: 'Test Title' }] });
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

  it('should hide project breadcrumb if isProjectRoute is false', async () => {
    const mockProjects: Project[] = [{ id: '1', name: 'Project 1' } as Project];
    projectServiceSpy.getAll.and.returnValue(of(mockProjects));
    userServiceSpy.getActiveProjectId.and.returnValue('1');
    component.isProjectRoute = false;

    fixture.detectChanges();
    await fixture.whenStable();

    const breadcrumbs = fixture.nativeElement.querySelectorAll('.breadcrumbs li');
    expect(breadcrumbs.length).toBe(1);
    expect(breadcrumbs[0].textContent).toContain('Test Title');
    expect(breadcrumbs[0].textContent).not.toContain('Project 1');
  });
});
