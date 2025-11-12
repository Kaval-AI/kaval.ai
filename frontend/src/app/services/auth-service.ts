import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';
import { tap } from 'rxjs/operators';


@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private loggedIn: boolean = false;
  private userDetails$ = new BehaviorSubject<UserDetails | null>(null);

  public userDetails = this.userDetails$.asObservable();

  constructor(private http: HttpClient) { }

  login() {
    window.location.href = "/api/login";
  }

  logout(): void {
    this.http.get("/api/logout").subscribe({
      next: () => {
        this.loggedIn = false;
        this.userDetails$.next(null);
      },
      error: () => {
        this.loggedIn = false;
        this.userDetails$.next(null);
      }
    });
  }

  updateUserDetails(): void {
    this.http.get<UserDetails>("/api/user/get_details").subscribe({
      next: (user_details) => {
        this.loggedIn = true;
        this.userDetails$.next(user_details as UserDetails);
      },
      error: () => {
        this.loggedIn = false;
        this.userDetails$.next(null);
      }
    });
  }

  getIsLoggedIn(): boolean {
    return this.loggedIn;
  }
}
