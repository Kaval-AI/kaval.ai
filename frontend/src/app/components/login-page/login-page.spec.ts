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
import { UserService } from '../../services/user-service';

import { LoginPage } from './login-page';

describe('LoginPage', () => {
  let component: LoginPage;
  let fixture: ComponentFixture<LoginPage>;
  let userService: UserService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LoginPage],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        UserService,
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(LoginPage);
    component = fixture.componentInstance;
    userService = TestBed.inject(UserService);
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should call userService.login when login is called', () => {
    const loginSpy = spyOn(userService, 'login');
    component.login();
    expect(loginSpy).toHaveBeenCalled();
  });

  it('should have header links', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const links = compiled.querySelectorAll('div.navbar a');
    expect(links.length).toBe(3);
    expect(links[0].textContent).toContain('Docs');
    expect(links[0].getAttribute('href')).toBe('https://docs.kaval.ai');
    expect(links[1].textContent).toContain('GitHub');
    expect(links[1].getAttribute('href')).toBe('https://github.com/kavalai/kavalai');
    expect(links[2].textContent).toContain('PyPI');
    expect(links[2].getAttribute('href')).toBe('https://pypi.org/project/kavalai/');
  });

  it('should have a login button', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const loginButton = compiled.querySelector('button.btn-primary');
    expect(loginButton).toBeTruthy();
    expect(loginButton?.textContent).toContain('Sign in with Google');
  });

  it('should call login when login button is clicked', () => {
    spyOn(component, 'login');
    const compiled = fixture.nativeElement as HTMLElement;
    const loginButton = compiled.querySelector('button.btn-primary') as HTMLButtonElement;
    loginButton.click();
    expect(component.login).toHaveBeenCalled();
  });

  it('should have a background container with the correct class', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const backgroundContainer = compiled.querySelector('.login-background-container');
    expect(backgroundContainer).toBeTruthy();
  });
});
