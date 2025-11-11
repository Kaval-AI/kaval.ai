import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatMessage } from './chat-message';

describe('Chatmessage', () => {
  let component: ChatMessage;
  let fixture: ComponentFixture<ChatMessage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatMessage]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChatMessage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
