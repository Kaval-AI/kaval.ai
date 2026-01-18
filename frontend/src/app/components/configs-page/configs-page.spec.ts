import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { ConfigsPage } from './configs-page';

describe('ConfigsPage', () => {
  let component: ConfigsPage;
  let fixture: ComponentFixture<ConfigsPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ConfigsPage],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(ConfigsPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
