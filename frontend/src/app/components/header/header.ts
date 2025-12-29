import { Component } from '@angular/core';
import { UserInfo } from '../user-info/user-info';
import { ProjectSelector } from '../project-selector/project-selector';

@Component({
  selector: 'app-header',
  imports: [UserInfo, ProjectSelector],
  templateUrl: './header.html',
  styleUrl: './header.css'
})
export class Header {
}
