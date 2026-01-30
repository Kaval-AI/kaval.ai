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
import { UserEditPage } from './user-edit-page';
import { UserService } from '../../services/user-service';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter, ActivatedRoute, Router } from '@angular/router';
import { of } from 'rxjs';
import { UserDetails } from '../../models/user-details';

describe('UserEditPage', () => {
  let component: UserEditPage;
  let fixture: ComponentFixture<UserEditPage>;
  let userService: UserService;
  let router: Router;

  const mockUsers: UserDetails[] = [
    { id: '123', name: 'John Doe', email: 'john@example.com', is_admin: false, picture: '' } as UserDetails,
    { id: '456', name: 'Admin', email: 'admin@example.com', is_admin: true, picture: '' } as UserDetails,
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UserEditPage],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        {
          provide: ActivatedRoute,
          useValue: {
            snapshot: {
              paramMap: {
                get: (key: string) => (key === 'id' ? '123' : null),
              },
            },
          },
        },
        UserService
      ]
    }).compileComponents();

    userService = TestBed.inject(UserService);
    router = TestBed.inject(Router);
    fixture = TestBed.createComponent(UserEditPage);
    component = fixture.componentInstance;

    spyOn(userService, 'getUsers').and.returnValue(of(mockUsers));
    spyOn(userService, 'getIsAdmin').and.returnValue(false);
    spyOn(userService, 'getUserDetailsValue').and.returnValue(mockUsers[0]);
  });

  it('should create and load user details', () => {
    fixture.detectChanges(); // ngOnInit
    expect(component.userId).toBe('123');
    expect(component.userDetails?.name).toBe('John Doe');
    expect(component.userForm.get('name')?.value).toBe('John Doe');
  });

  it('should disable email and is_admin if not admin', () => {
    fixture.detectChanges();
    expect(component.userForm.get('email')?.disabled).toBeTrue();
    expect(component.userForm.get('is_admin')?.disabled).toBeTrue();
  });

  it('should allow editing email and is_admin if admin', () => {
    (userService.getIsAdmin as jasmine.Spy).and.returnValue(true);
    fixture.detectChanges();
    expect(component.userForm.get('email')?.enabled).toBeTrue();
    expect(component.userForm.get('is_admin')?.enabled).toBeTrue();
  });

  it('should navigate back on cancel', () => {
    const navigateSpy = spyOn(router, 'navigate');
    component.cancel();
    expect(navigateSpy).toHaveBeenCalledWith(['/']);
  });

  it('should save user and navigate on saveUser', () => {
    const updateSpy = spyOn(userService, 'updateUser').and.returnValue(of({} as any));
    const navigateSpy = spyOn(router, 'navigate');

    fixture.detectChanges();
    component.userForm.patchValue({ name: 'Jane Doe' });
    component.saveUser();

    expect(updateSpy).toHaveBeenCalled();
    expect(navigateSpy).toHaveBeenCalledWith(['/']);
  });
});
