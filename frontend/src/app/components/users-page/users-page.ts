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
  showAddModal = false;
  showEditModal = false;
  showDeleteModal = false;
  showErrorModal = false;
  errorMessage = '';
  userToDelete: UserDetails | null = null;
  selectedUser: UserDetails | null = null;
  isSaving = false;
  isDeleting = false;

  userForm = this.fb.group({
    email: ['', [Validators.required, Validators.email]],
    name: ['', [Validators.required]],
    is_admin: [false],
    picture: [''],
  });

  editForm = this.fb.group({
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
    this.selectedUser = user;
    this.editForm.patchValue(user);
    this.showEditModal = true;
  }

  addUser() {
    this.showAddModal = true;
    this.userForm.reset({ is_admin: false });
  }

  closeAddModal() {
    this.showAddModal = false;
    this.userForm.reset();
  }

  closeEditModal() {
    this.showEditModal = false;
    this.selectedUser = null;
    this.editForm.reset();
  }

  saveUser() {
    if (this.userForm.invalid) return;

    this.isSaving = true;
    const userData = this.userForm.value as Partial<UserDetails>;

    this.userService.createUser(userData).subscribe({
      next: () => {
        this.loadUsers();
        this.showAddModal = false;
        this.isSaving = false;
      },
      error: (err) => {
        this.isSaving = false;
        this.showError(err.error?.message || 'Failed to create user.');
      }
    });
  }

  updateUser() {
    if (this.editForm.invalid || !this.selectedUser?.id) return;

    this.isSaving = true;
    const userData = this.editForm.value as Partial<UserDetails>;

    this.userService.updateUser(this.selectedUser.id, userData).subscribe({
      next: () => {
        this.loadUsers();
        this.showEditModal = false;
        this.isSaving = false;
        // If it's self, update details in service
        const currentUser = this.userService.getUserDetailsValue();
        if (currentUser && currentUser.id === this.selectedUser?.id) {
          this.userService.updateUserDetails();
        }
      },
      error: (err) => {
        this.isSaving = false;
        this.showError(err.error?.message || 'Failed to update user.');
      }
    });
  }

  deleteUser(user: UserDetails) {
    if (!user.id) return;

    // Safety check: don't allow deleting yourself
    const currentUser = this.userService.getUserDetailsValue();
    if (currentUser && currentUser.id === user.id) {
      this.showError('You cannot delete yourself.');
      return;
    }

    this.userToDelete = user;
    this.showDeleteModal = true;
  }

  confirmDelete() {
    if (!this.userToDelete?.id) return;

    this.isDeleting = true;
    this.userService.deleteUser(this.userToDelete.id).subscribe({
      next: () => {
        this.loadUsers();
        this.closeDeleteModal();
        this.isDeleting = false;
      },
      error: (err) => {
        this.isDeleting = false;
        this.showError(err.error?.message || 'Failed to delete user.');
      }
    });
  }

  closeDeleteModal() {
    this.showDeleteModal = false;
    this.userToDelete = null;
  }

  showError(message: string) {
    this.errorMessage = message;
    this.showErrorModal = true;
  }

  closeErrorModal() {
    this.showErrorModal = false;
    this.errorMessage = '';
  }
}
