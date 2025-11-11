import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DeleteChatDialog } from './delete-chat-dialog';

describe('DeleteChatDialog', () => {
  let component: DeleteChatDialog;
  let fixture: ComponentFixture<DeleteChatDialog>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DeleteChatDialog]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DeleteChatDialog);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
