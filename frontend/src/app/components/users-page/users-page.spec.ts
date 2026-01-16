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

  it('should navigate to user-edit when editUser is called', () => {
    const mockUser = { id: '123', email: 'test@example.com' };
    const navigateSpy = spyOn(router, 'navigate');

    component.editUser(mockUser as any);

    expect(navigateSpy).toHaveBeenCalledWith(['/user-edit', '123']);
  });
});
