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
import { Router } from '@angular/router';
import { of } from 'rxjs';

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
    userServiceSpy = jasmine.createSpyObj('UserService', ['getActiveProjectId', 'setActiveProject', 'updateUserDetails'], { userDetails: of({ id: 'u1' }) });
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
        { provide: ProjectService, useValue: projectServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        { provide: NavigationService, useValue: navigationServiceSpy },
        { provide: Router, useValue: routerSpy },
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(Header);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should load projects on init', () => {
    const mockProjects: Project[] = [{ id: '1', name: 'Project 1' } as Project];
    projectServiceSpy.getAll.and.returnValue(of(mockProjects));

    component.ngOnInit();

    expect(projectServiceSpy.getAll).toHaveBeenCalled();
    expect(component.projects).toEqual(mockProjects);
  });

  it('should set active project if none selected and projects exist', () => {
    const mockProjects: Project[] = [{ id: '1', name: 'Project 1' } as Project];
    projectServiceSpy.getAll.and.returnValue(of(mockProjects));
    userServiceSpy.getActiveProjectId.and.returnValue(null);

    component.loadProjects();

    expect(userServiceSpy.setActiveProject).toHaveBeenCalledWith('1');
  });

  it('should handle project selection and refresh current component', () => {
    const event = { target: { value: '2' } } as any;
    component.onProjectSelect(event);
    expect(userServiceSpy.setActiveProject).toHaveBeenCalledWith('2');
    expect(routerSpy.navigateByUrl).toHaveBeenCalledWith('/', { skipLocationChange: true });
  });
});
