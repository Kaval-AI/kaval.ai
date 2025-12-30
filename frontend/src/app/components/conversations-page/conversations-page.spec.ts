import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ConversationsPage } from './conversations-page';

describe('ConversationsPage', () => {
  let component: ConversationsPage;
  let fixture: ComponentFixture<ConversationsPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ConversationsPage]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ConversationsPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
