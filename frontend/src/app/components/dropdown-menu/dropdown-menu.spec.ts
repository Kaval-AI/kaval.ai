/*
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import { Component, TemplateRef, ViewChild } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DropdownMenuTriggerDirective } from './dropdown-menu';
import { By } from '@angular/platform-browser';
import { OverlayContainer } from '@angular/cdk/overlay';

@Component({
  standalone: true,
  imports: [DropdownMenuTriggerDirective],
  template: `
    <button id="trigger" [dropdownMenuTrigger]="menu">Open</button>
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
  let overlayContainer: OverlayContainer;
  let overlayContainerElement: HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent, DropdownMenuTriggerDirective],
    }).compileComponents();

    fixture = TestBed.createComponent(TestHostComponent);
    component = fixture.componentInstance;
    overlayContainer = TestBed.inject(OverlayContainer);
    overlayContainerElement = overlayContainer.getContainerElement();
    fixture.detectChanges();
  });

  afterEach(() => {
    overlayContainer.ngOnDestroy();
  });

  it('should create host component', () => {
    expect(component).toBeTruthy();
  });

  it('should open and close menu on click', () => {
    const trigger = fixture.debugElement.query(By.css('#trigger')).nativeElement;

    // Open
    trigger.click();
    fixture.detectChanges();
    expect(overlayContainerElement.querySelector('#test-menu')).toBeTruthy();

    // Close
    trigger.click();
    fixture.detectChanges();
    expect(overlayContainerElement.querySelector('#test-menu')).toBeFalsy();
  });

  it('should close menu on backdrop click', () => {
    const trigger = fixture.debugElement.query(By.css('#trigger')).nativeElement;

    // Open
    trigger.click();
    fixture.detectChanges();
    expect(overlayContainerElement.querySelector('#test-menu')).toBeTruthy();

    // Backdrop click
    const backdrop = overlayContainerElement.querySelector('.cdk-overlay-backdrop') as HTMLElement;
    backdrop.click();
    fixture.detectChanges();
    expect(overlayContainerElement.querySelector('#test-menu')).toBeFalsy();
  });
});
