import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RenameChatDialog } from './rename-chat-dialog';

describe('RenameChatDialog', () => {
  let component: RenameChatDialog;
  let fixture: ComponentFixture<RenameChatDialog>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RenameChatDialog]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RenameChatDialog);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
