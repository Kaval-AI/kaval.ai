import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatUserInput } from './chat-user-input';

describe('ChatUserInput', () => {
  let component: ChatUserInput;
  let fixture: ComponentFixture<ChatUserInput>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatUserInput],
    }).compileComponents();

    fixture = TestBed.createComponent(ChatUserInput);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
