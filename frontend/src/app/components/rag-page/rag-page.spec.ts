import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { RagPage } from './rag-page';

describe('RagPage', () => {
  let component: RagPage;
  let fixture: ComponentFixture<RagPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RagPage],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(RagPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
