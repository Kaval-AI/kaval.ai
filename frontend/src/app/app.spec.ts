import { TestBed } from '@angular/core/testing';
import { App } from './app';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { UserService } from './services/user-service';

describe('App', () => {
  let userService: UserService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        UserService
      ]
    }).compileComponents();

    userService = TestBed.inject(UserService);
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should call updateUserDetails on init', () => {
    const spy = spyOn(userService, 'updateUserDetails');
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    expect(spy).toHaveBeenCalled();
  });

  it('should return isLoggedIn from userService', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    spyOn(userService, 'getIsLoggedIn').and.returnValue(true);
    expect(app.isLoggedIn).toBeTrue();
  });
});
