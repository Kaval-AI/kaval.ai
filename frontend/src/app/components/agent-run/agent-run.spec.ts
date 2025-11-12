import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AgentRun } from './agent-run';

describe('AgentRun', () => {
  let component: AgentRun;
  let fixture: ComponentFixture<AgentRun>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AgentRun]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AgentRun);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
