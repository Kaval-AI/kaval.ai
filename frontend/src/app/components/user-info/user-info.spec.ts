import { ComponentFixture, TestBed } from '@angular/core/testing';
import { UserInfo } from './user-info';
import { provideHttpClient } from '@angular/common/http';

describe('UserInfo', () => {
  let component: UserInfo;
  let fixture: ComponentFixture<UserInfo>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UserInfo],
      providers: [provideHttpClient()]
    }).compileComponents();

    fixture = TestBed.createComponent(UserInfo);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
