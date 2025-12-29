import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Userinfo } from './user-info';

describe('Userinfo', () => {
  let component: Userinfo;
  let fixture: ComponentFixture<Userinfo>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Userinfo]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Userinfo);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
