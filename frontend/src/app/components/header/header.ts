import { Component } from '@angular/core';
import { UserInfo } from '../user-info/user-info';

@Component({
  selector: 'app-header',
  imports: [UserInfo],
  templateUrl: './header.html',
  styleUrl: './header.css',
})
export class Header {}
