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

import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ToastService } from '../../services/toast-service';

@Component({
  selector: 'app-toast',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="z-[9999] pointer-events-none fixed inset-0">
      @for (toast of toastService.toasts(); track toast.id) {
        <div class="alert absolute transition-all duration-300 pointer-events-auto"
          [style.left.px]="toast.position?.x"
          [style.top.px]="toast.position?.y"
          [ngClass]="{
            'alert-success': toast.type === 'success',
            'alert-error': toast.type === 'error',
            'alert-info': toast.type === 'info',
            'alert-warning': toast.type === 'warning',
            'toast toast-end toast-bottom relative': !toast.position
          }">
          <span>{{ toast.message }}</span>
        </div>
      }
    </div>
  `,
})
export class Toast {
  public toastService = inject(ToastService);
}
