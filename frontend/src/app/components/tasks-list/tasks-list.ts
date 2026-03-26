/*
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Task } from '../../models/task';
import { JsonTreeComponent } from '../json-tree/json-tree';

@Component({
  selector: 'app-tasks-list',
  standalone: true,
  imports: [CommonModule, JsonTreeComponent],
  templateUrl: './tasks-list.html',
  styleUrl: './tasks-list.css',
})
export class TasksList {
  @Input() tasks: Task[] = [];
  @Input() runId: string = '';

  formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }
}
