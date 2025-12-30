import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TestsPage } from './tests-page';

describe('TestsPage', () => {
  let component: TestsPage;
  let fixture: ComponentFixture<TestsPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestsPage]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TestsPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
