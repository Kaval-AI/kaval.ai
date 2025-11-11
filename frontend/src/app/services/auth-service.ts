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

  login(email: string, password: string): Observable<any> {
    var credentials = {
      "Email": email,
      "Password": password
    }
    return this.http.post("/api/user/login", credentials).pipe(
      tap(() => {
        console.log("Login successful");
        this.loggedIn = true;
      })
    );
  }

  logout(): void {
    this.http.post("/api/user/logout", {}).subscribe({
      next: () => this.loggedIn = false,
    });
  }

  updateLoginStatus(): void {
    this.http.get<{ isLoggedIn: boolean }>("/api/user/isloggedin").subscribe({
      next: (result) => {
        const isLoggedIn = result["isLoggedIn"];
        console.log("Logged in status:", isLoggedIn);
        this.loggedIn = isLoggedIn
      },
      error: () => {
        this.loggedIn = false;
      }
    });
  }

  getIsLoggedIn(): boolean {
    return this.loggedIn;
  }

  updateUserDetails(): void {
    if (!this.loggedIn) {
      this.userDetails$.next(null);
    }
    this.http.get<UserDetails>("/api/user/details").subscribe({
      next: (details) => {
        this.userDetails$.next(details);
      },
      error: () => {
        this.userDetails$.next(null);
      }
    });
  }
}
