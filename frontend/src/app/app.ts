import { Component, OnInit } from '@angular/core';
import { LoginPage } from './components/login-page/login-page';
import { AuthService } from './services/auth-service';
import { SidebarMenu } from './components/sidebar-menu/sidebar-menu';
import { UserInfo } from './components/user-info/user-info';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [LoginPage, SidebarMenu, UserInfo, RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  constructor(private authService: AuthService) {
  }

  ngOnInit(): void {
    this.authService.updateUserDetails();
  }

  get isLoggedIn(): boolean {
    return this.authService.getIsLoggedIn();
  }
}
