import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { DropdownMenuTriggerDirective } from '../dropdown-menu/dropdown-menu';
import { MatButtonModule } from '@angular/material/button';
import { UserDetails } from '../../models/user-details';
import { UserService } from '../../services/user-service';

@Component({
  selector: 'app-user-info',
  templateUrl: './user-info.html',
  styleUrls: ['./user-info.css'],
  imports: [CommonModule, DropdownMenuTriggerDirective],
})
export class UserInfo implements OnInit {
  userDetails: UserDetails | null = null;

  constructor(private userService: UserService) {}

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
}
