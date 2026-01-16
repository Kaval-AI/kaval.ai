import { ComponentFixture, TestBed } from '@angular/core/testing';
import { UsersPage } from './users-page';
import { UserService } from '../../services/user-service';
import { of } from 'rxjs';

describe('UsersPage', () => {
  let component: UsersPage;
  let fixture: ComponentFixture<UsersPage>;
  let userServiceSpy: jasmine.SpyObj<UserService>;

  beforeEach(async () => {
    userServiceSpy = jasmine.createSpyObj('UserService', ['getUsers', 'getUserDetailsValue']);
    userServiceSpy.getUsers.and.returnValue(of([]));

    await TestBed.configureTestingModule({
      imports: [UsersPage],
      providers: [
        { provide: UserService, useValue: userServiceSpy }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(UsersPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
