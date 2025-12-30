import {
  Directive,
  Input,
  TemplateRef,
  ViewContainerRef,
  HostListener,
} from '@angular/core';
import { Overlay, OverlayRef } from '@angular/cdk/overlay';
import { TemplatePortal } from '@angular/cdk/portal';

@Directive({
  selector: '[dropdownMenuTrigger]',
  standalone: true,
})
export class DropdownMenuTriggerDirective {
  @Input('dropdownMenuTrigger') menuTemplate!: TemplateRef<any>;
  private overlayRef: OverlayRef | null = null;

  constructor(
    private overlay: Overlay,
    private vcr: ViewContainerRef
  ) {}

  @HostListener('click')
  toggleMenu() {
    if (this.overlayRef) {
      this.closeMenu();
    } else {
      this.openMenu();
    }
  }

  private openMenu() {
    // 1. Create Position Strategy (Where it sits)
    const positionStrategy = this.overlay
      .position()
      .flexibleConnectedTo(this.vcr.element)
      .withPositions([
        {
          originX: 'start',
          originY: 'bottom',
          overlayX: 'start',
          overlayY: 'top',
          offsetY: 0,
        },
        {
          originX: 'start',
          originY: 'top',
          overlayX: 'start',
          overlayY: 'bottom',
          offsetY: 0,
        },
      ]);

    // 2. Create Overlay Configuration
    this.overlayRef = this.overlay.create({
      positionStrategy,
      hasBackdrop: true,
      backdropClass: 'cdk-overlay-transparent-backdrop',
      scrollStrategy: this.overlay.scrollStrategies.reposition(),
    });

    // 3. Close menu when backdrop is clicked
    this.overlayRef.backdropClick().subscribe(() => this.closeMenu());

    // 4. Attach the template to the overlay
    const portal = new TemplatePortal(this.menuTemplate, this.vcr);
    this.overlayRef.attach(portal);
  }

  private closeMenu() {
    if (this.overlayRef) {
      this.overlayRef.detach();
      this.overlayRef = null;
    }
  }
}
