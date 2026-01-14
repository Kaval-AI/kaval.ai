import { Component, OnInit } from '@angular/core';
import { LoginPage } from './components/login-page/login-page';
import { UserService } from './services/user-service';
import { SidebarMenu } from './components/sidebar-menu/sidebar-menu';
import { UserInfo } from './components/user-info/user-info';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [LoginPage, SidebarMenu, UserInfo, RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit {
  constructor(private userService: UserService) {}

  ngOnInit(): void {
    this.userService.updateUserDetails();
  }

  get isLoggedIn(): boolean {
    return this.userService.getIsLoggedIn();
  }
}
