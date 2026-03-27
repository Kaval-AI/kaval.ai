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

import { Component, inject, ViewChild, ElementRef } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';
import { UserService } from '../../../services/user-service';

@Component({
  selector: 'app-header-dropdown',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  templateUrl: './header-dropdown.html',
  styleUrl: './header-dropdown.css',
})
export class HeaderDropdown {
  @ViewChild('dropdownButton') dropdownButton!: ElementRef<HTMLElement>;

  isMenuOpen = false;

  private userService = inject(UserService);

  isAdmin(): boolean {
    return this.userService.getIsAdmin();
  }

  openMenu(): void {
    this.isMenuOpen = true;
  }

  closeMenu(): void {
    setTimeout(() => {
      this.isMenuOpen = false;
    }, 200);
  }

  onMenuClick(): void {
    this.closeMenu();
  }
}
