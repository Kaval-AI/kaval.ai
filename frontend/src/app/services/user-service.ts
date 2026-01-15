import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import { UserDetails } from '../models/user-details';

@Injectable({
  providedIn: 'root',
})
export class UserService {
  private loggedIn: boolean = false;
  private userDetails$ = new BehaviorSubject<UserDetails | null>(null);

  public userDetails = this.userDetails$.asObservable();

  constructor(private http: HttpClient) {}

  login() {
    this.redirect('/api/login');
  }

  private redirect(url: string) {
    window.location.href = url;
  }

  logout(): void {
    this.http.get('/api/logout').subscribe({
      next: () => {
        this.loggedIn = false;
        this.userDetails$.next(null);
      },
      error: () => {
        this.loggedIn = false;
        this.userDetails$.next(null);
      },
    });
  }

  updateUserDetails(): void {
    this.http.get<UserDetails>('/api/user/get_details').subscribe({
      next: (user_details) => {
        this.loggedIn = true;
        this.userDetails$.next(user_details as UserDetails);
      },
      error: () => {
        this.loggedIn = false;
        this.userDetails$.next(null);
      },
    });
  }

  getIsLoggedIn(): boolean {
    return this.loggedIn;
  }

  getIsAdmin(): boolean {
    const user = this.userDetails$.value;
    return this.loggedIn && user != null && user.is_admin;
  }

  getActiveProjectId(): string | null {
    const user = this.userDetails$.value;
    if (user && user.active_project_id) {
      return user.active_project_id;
    }
    return null;
  }

  setActiveProject(projectId: string): void { // Changed from Observable<any> to void
    this.http.post<any>(`/api/user/set_active_project/${projectId}`, {}).subscribe({
      next: () => {
        const currentDetails = this.userDetails$.value;
        if (currentDetails) {
          this.userDetails$.next({
            ...currentDetails,
            active_project_id: projectId
          });
        }
      },
      error: (err) => {
        console.error('Failed to set active project', err);
      }
    });
  }
}
