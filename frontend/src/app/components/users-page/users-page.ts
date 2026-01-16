import { Component, OnInit, inject } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { UserService } from '../../services/user-service';
import { UserDetails } from '../../models/user-details';

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

  users: UserDetails[] = [];
  selectedUser: UserDetails | null = null;
  isEditing = false;
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
    this.selectedUser = user;
    this.isEditing = true;
    this.showForm = true;
    this.userForm.patchValue(user);
  }

  addUser() {
    this.selectedUser = null;
    this.isEditing = false;
    this.showForm = true;
    this.userForm.reset({ is_admin: false });
  }

  cancelForm() {
    this.showForm = false;
    this.selectedUser = null;
    this.userForm.reset();
  }

  saveUser() {
    if (this.userForm.invalid) return;

    const userData = this.userForm.value as Partial<UserDetails>;

    if (this.isEditing && this.selectedUser?.id) {
      this.userService.updateUser(this.selectedUser.id, userData).subscribe(() => {
        this.loadUsers();
        this.showForm = false;
      });
    } else {
      this.userService.createUser(userData).subscribe(() => {
        this.loadUsers();
        this.showForm = false;
      });
    }
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
