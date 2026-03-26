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
import { UserInfo } from './user-info';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { UserService } from '../../services/user-service';
import { of } from 'rxjs';
import { UserDetails } from '../../models/user-details';
import { provideRouter } from '@angular/router';
import { Router } from '@angular/router';

describe('UserInfo', () => {
  let component: UserInfo;
  let fixture: ComponentFixture<UserInfo>;
  let userService: UserService;
  let router: Router;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UserInfo],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        UserService
      ]
    }).compileComponents();

    userService = TestBed.inject(UserService);
    router = TestBed.inject(Router);
    fixture = TestBed.createComponent(UserInfo);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should update userDetails when service emits', () => {
    const mockUser: UserDetails = { name: 'Test User' } as UserDetails;
    // Call ngOnInit indirectly via detectChanges
    fixture.detectChanges();

    // Use a private access to next for the subject if possible, or just mock the HTTP call
    // Actually, since we use the real service, we can mock the HTTP call that updateUserDetails makes
    const httpTestingController = TestBed.inject(HttpTestingController);
    const req = httpTestingController.expectOne('/api/user/get_details');
    req.flush(mockUser);

    expect(component.userDetails).toEqual(mockUser);
  });

  it('should call userService.logout when logout is called', () => {
    fixture.detectChanges();
    const logoutSpy = spyOn(userService, 'logout');
    component.logout();
    expect(logoutSpy).toHaveBeenCalled();
  });

  it('should navigate to user-edit when editProfile is called', () => {
    const mockUser: UserDetails = { id: '123', name: 'Test User' } as UserDetails;
    component.userDetails = mockUser;
    const navigateSpy = spyOn(router, 'navigate');

    component.editProfile();

    expect(navigateSpy).toHaveBeenCalledWith(['/user-edit', '123']);
  });

  it('should open menu when openMenu is called', () => {
    component.openMenu();
    expect(component.isMenuOpen).toBeTrue();
  });

  it('should close menu when closeMenu is called', (done) => {
    component.isMenuOpen = true;
    component.closeMenu();
    setTimeout(() => {
      expect(component.isMenuOpen).toBeFalse();
      done();
    }, 250);
  });

  it('should close menu when onMenuClick is called', () => {
    component.isMenuOpen = true;
    const closeMenuSpy = spyOn(component, 'closeMenu');
    component.onMenuClick();
    expect(closeMenuSpy).toHaveBeenCalled();
  });
});
