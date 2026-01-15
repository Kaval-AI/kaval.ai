import { ComponentFixture, TestBed } from '@angular/core/testing';
import { UserInfo } from './user-info';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { UserService } from '../../services/user-service';
import { of } from 'rxjs';
import { UserDetails } from '../../models/user-details';

describe('UserInfo', () => {
  let component: UserInfo;
  let fixture: ComponentFixture<UserInfo>;
  let userService: UserService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UserInfo],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        UserService
      ]
    }).compileComponents();

    userService = TestBed.inject(UserService);
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
});
