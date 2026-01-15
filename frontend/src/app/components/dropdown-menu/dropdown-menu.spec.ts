import { Component, TemplateRef, ViewChild } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DropdownMenuTriggerDirective } from './dropdown-menu';

@Component({
  standalone: true,
  imports: [DropdownMenuTriggerDirective],
  template: `
    <button [dropdownMenuTrigger]="menu">Open</button>
    <ng-template #menu>
      <div id="test-menu">Menu Content</div>
    </ng-template>
  `,
})
class TestHostComponent {
  @ViewChild('menu') menu!: TemplateRef<any>;
}

describe('DropdownMenuTriggerDirective', () => {
  let component: TestHostComponent;
  let fixture: ComponentFixture<TestHostComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent, DropdownMenuTriggerDirective],
    }).compileComponents();

    fixture = TestBed.createComponent(TestHostComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create host component', () => {
    expect(component).toBeTruthy();
  });
});
