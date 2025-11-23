import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SystemPrompt } from './system-prompt';

describe('SystemPrompt', () => {
  let component: SystemPrompt;
  let fixture: ComponentFixture<SystemPrompt>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SystemPrompt]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SystemPrompt);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
