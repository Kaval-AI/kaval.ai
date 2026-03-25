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
import { UsersPage } from './users-page';
import { UserService } from '../../services/user-service';
import { of } from 'rxjs';
import { provideRouter, Router } from '@angular/router';

describe('UsersPage', () => {
  let component: UsersPage;
  let fixture: ComponentFixture<UsersPage>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let router: Router;

  beforeEach(async () => {
    userServiceSpy = jasmine.createSpyObj('UserService', ['getUsers', 'getUserDetailsValue']);
    userServiceSpy.getUsers.and.returnValue(of([]));

    await TestBed.configureTestingModule({
      imports: [UsersPage],
      providers: [
        { provide: UserService, useValue: userServiceSpy },
        provideRouter([])
      ]
    }).compileComponents();

    router = TestBed.inject(Router);
    fixture = TestBed.createComponent(UsersPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should open edit modal when editUser is called', () => {
    const mockUser = { id: '123', email: 'test@example.com', name: 'Test User', is_admin: false, picture: '' };

    component.editUser(mockUser as any);

    expect(component.selectedUser).toBe(mockUser as any);
    expect(component.showEditModal).toBeTrue();
    expect(component.editForm.value.email).toBe('test@example.com');
  });

  it('should open delete modal when deleteUser is called', () => {
    const mockUser = { id: '123', email: 'test@example.com', name: 'Test User', is_admin: false, picture: '' };
    userServiceSpy.getUserDetailsValue.and.returnValue({ id: '456' } as any);

    component.deleteUser(mockUser as any);

    expect(component.userToDelete).toBe(mockUser as any);
    expect(component.showDeleteModal).toBeTrue();
  });

  it('should show error modal when trying to delete self', () => {
    const mockUser = { id: '123', email: 'test@example.com', name: 'Test User', is_admin: false, picture: '' };
    userServiceSpy.getUserDetailsValue.and.returnValue({ id: '123' } as any);

    component.deleteUser(mockUser as any);

    expect(component.showErrorModal).toBeTrue();
    expect(component.errorMessage).toBe('You cannot delete yourself.');
  });

  it('should display page title and description', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('System Users');
    expect(compiled.querySelector('p')?.textContent).toContain('Manage system users and their permissions.');
  });

  it('should use non-outline button for edit', () => {
    const mockUser = { id: '123', email: 'test@example.com', name: 'Test User', is_admin: false, picture: '' };
    component.users = [mockUser as any];
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const editButton = compiled.querySelector('button[title="Edit"]');
    expect(editButton).toBeTruthy();
    expect(editButton?.classList.contains('btn-outline')).toBeFalse();
    expect(editButton?.classList.contains('btn-primary')).toBeTrue();
  });
});
