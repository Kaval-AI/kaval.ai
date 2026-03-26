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
import { CommonModule } from '@angular/common';
import { Component, OnInit, inject, ViewChild, ElementRef } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { UserDetails } from '../../models/user-details';
import { UserService } from '../../services/user-service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-user-info',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  templateUrl: './user-info.html',
  styleUrls: ['./user-info.css'],
})
export class UserInfo implements OnInit {
  @ViewChild('dropdownButton') dropdownButton!: ElementRef<HTMLElement>;

  private userService = inject(UserService);
  private router = inject(Router);

  userDetails: UserDetails | null = null;
  isMenuOpen = false;

  ngOnInit(): void {
    this.userService.userDetails.subscribe((details: UserDetails | null) => {
      this.userDetails = details;
    });
    this.userService.updateUserDetails();
  }

  logout(): void {
    this.userService.logout();
  }

  editProfile(): void {
    if (this.userDetails?.id) {
      this.router.navigate(['/user-edit', this.userDetails.id]);
    }
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
