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
import { UserService } from '../../services/user-service';
import { UserDetails } from '../../models/user-details';
import { Router } from '@angular/router';

@Component({
  selector: 'app-users-page',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule],
  templateUrl: './users-page.html',
  styleUrl: './users-page.css',
})
export class UsersPage implements OnInit {
  private userService = inject(UserService);
  private fb = inject(FormBuilder);
  private router = inject(Router);

  users: UserDetails[] = [];
  showForm = false;

  userForm = this.fb.group({
    email: ['', [Validators.required, Validators.email]],
    name: ['', [Validators.required]],
    is_admin: [false],
    picture: [''],
  });

  ngOnInit() {
    this.loadUsers();
  }

  loadUsers() {
    this.userService.getUsers().subscribe((data) => {
      this.users = data;
    });
  }

  editUser(user: UserDetails) {
    if (user.id) {
      this.router.navigate(['/user-edit', user.id]);
    }
  }

  addUser() {
    this.showForm = true;
    this.userForm.reset({ is_admin: false });
  }

  cancelForm() {
    this.showForm = false;
    this.userForm.reset();
  }

  saveUser() {
    if (this.userForm.invalid) return;

    const userData = this.userForm.value as Partial<UserDetails>;

    this.userService.createUser(userData).subscribe(() => {
      this.loadUsers();
      this.showForm = false;
    });
  }

  deleteUser(user: UserDetails) {
    if (!user.id) return;

    // Safety check: don't allow deleting yourself
    const currentUser = this.userService.getUserDetailsValue();
    if (currentUser && currentUser.id === user.id) {
      alert('You cannot delete yourself.');
      return;
    }

    if (confirm(`Are you sure you want to delete user ${user.email}?`)) {
      this.userService.deleteUser(user.id).subscribe(() => {
        this.loadUsers();
      });
    }
  }
}
