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
import { Component, OnInit, inject } from '@angular/core';
import { DropdownMenuTriggerDirective } from '../dropdown-menu/dropdown-menu';
import { UserDetails } from '../../models/user-details';
import { UserService } from '../../services/user-service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-user-info',
  templateUrl: './user-info.html',
  styleUrls: ['./user-info.css'],
  imports: [CommonModule, DropdownMenuTriggerDirective],
})
export class UserInfo implements OnInit {
  private userService = inject(UserService);
  private router = inject(Router);

  userDetails: UserDetails | null = null;

  ngOnInit(): void {
    this.userService.userDetails.subscribe((details: UserDetails | null) => {
      console.log(details);
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
}
