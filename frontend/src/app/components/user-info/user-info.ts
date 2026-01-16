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
