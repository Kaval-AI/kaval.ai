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
import { Component, OnInit, inject } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { UserService } from '../../services/user-service';
import { UserDetails } from '../../models/user-details';

@Component({
  selector: 'app-user-edit-page',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule],
  templateUrl: './user-edit-page.html',
  styleUrl: './user-edit-page.css',
})
export class UserEditPage implements OnInit {
  private userService = inject(UserService);
  private fb = inject(FormBuilder);
  private route = inject(ActivatedRoute);
  private router = inject(Router);

  userId: string | null = null;
  userDetails: UserDetails | null = null;
  isAdmin = false;
  isSelf = false;

  userForm = this.fb.group({
    email: ['', [Validators.required, Validators.email]],
    name: ['', [Validators.required]],
    is_admin: [false],
    picture: [''],
  });

  ngOnInit() {
    this.isAdmin = this.userService.getIsAdmin();
    this.userId = this.route.snapshot.paramMap.get('id');

    const currentUser = this.userService.getUserDetailsValue();
    if (currentUser && currentUser.id === this.userId) {
      this.isSelf = true;
    }

    if (this.userId) {
      this.loadUser();
    }
  }

  loadUser() {
    // If it's self, we might already have it in UserService,
    // but better to fetch fresh from all users list if admin,
    // or we might need a getById in UserService.
    // Current UserService only has getUsers() (all).
    this.userService.getUsers().subscribe((users) => {
      const user = users.find(u => u.id === this.userId);
      if (user) {
        this.userDetails = user;
        this.userForm.patchValue(user);

        // Disable email and is_admin if not admin
        if (!this.isAdmin) {
          this.userForm.get('email')?.disable();
          this.userForm.get('is_admin')?.disable();
        }
      } else {
        // Handle user not found
        this.router.navigate(['/']);
      }
    });
  }

  saveUser() {
    if (this.userForm.invalid || !this.userId) return;

    const userData = this.userForm.getRawValue() as Partial<UserDetails>;

    this.userService.updateUser(this.userId, userData).subscribe(() => {
      if (this.isSelf) {
        this.userService.updateUserDetails();
      }
      this.router.navigate(['/']);
    });
  }

  cancel() {
    this.router.navigate(['/']);
  }
}
