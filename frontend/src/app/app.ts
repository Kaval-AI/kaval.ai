import { Component, OnInit } from '@angular/core';
import { Header } from './components/header/header';
import { Chat } from "./components/chat/chat";
import { LoginPage } from './components/login-page/login-page';
import { AuthService } from './services/auth-service';

@Component({
  selector: 'app-root',
  imports: [Header, Chat, LoginPage],
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
