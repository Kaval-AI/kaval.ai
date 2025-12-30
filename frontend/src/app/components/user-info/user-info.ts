import { Component, OnInit } from '@angular/core';
import { AuthService } from '../../services/auth-service';
import { MatButtonModule } from '@angular/material/button';
import { CommonModule } from '@angular/common';
import { DropdownMenuTriggerDirective } from '../dropdown-menu/dropdown-menu';

@Component({
  selector: 'app-user-info',
  templateUrl: './user-info.html',
  styleUrls: ['./user-info.css'],
  imports: [CommonModule, DropdownMenuTriggerDirective],
})
export class UserInfo implements OnInit {
  userDetails: UserDetails | null = null;

  constructor(private authService: AuthService) {}

  ngOnInit(): void {
    this.authService.userDetails.subscribe((details) => {
      console.log(details);
      this.userDetails = details;
    });
    this.authService.updateUserDetails();
  }

  logout(): void {
    this.authService.logout();
  }
}
