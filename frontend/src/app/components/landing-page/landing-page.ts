import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ProjectsPage } from '../projects-page/projects-page';
import { UserService } from '../../services/user-service';

@Component({
  selector: 'app-landing-page',
  standalone: true,
  imports: [CommonModule, ProjectsPage],
  templateUrl: './landing-page.html',
  styleUrl: './landing-page.css',
})
export class LandingPage {
  constructor(private userService: UserService) {}

  getIsAdmin(): boolean {
    return this.userService.getIsAdmin();
  }
}
