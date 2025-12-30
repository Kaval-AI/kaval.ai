import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RagPage } from './rag-page';

describe('RagPage', () => {
  let component: RagPage;
  let fixture: ComponentFixture<RagPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RagPage]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RagPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
