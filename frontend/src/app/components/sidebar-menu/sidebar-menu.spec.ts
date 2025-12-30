import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SidebarMenu } from './sidebar-menu';

describe('SidebarMenu', () => {
  let component: SidebarMenu;
  let fixture: ComponentFixture<SidebarMenu>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SidebarMenu]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SidebarMenu);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
