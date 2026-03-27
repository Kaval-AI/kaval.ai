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

import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Task } from '../../models/task';

interface TaskBox {
  id: string;
  name: string;
  left: number; // percentage
  width: number; // percentage
  startTime: number;
  duration: number;
}

@Component({
  selector: 'app-task-timeline-chart',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './task-timeline-chart.html',
  styleUrl: './task-timeline-chart.css',
})
export class TaskTimelineChart implements OnChanges {
  @Input() tasks: Task[] = [];

  taskBoxes: TaskBox[] = [];
  minTime: number = 0;
  maxTime: number = 0;
  totalDuration: number = 0;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['tasks']) {
      this.calculateTaskBoxes();
    }
  }

  private calculateTaskBoxes(): void {
    if (!this.tasks || this.tasks.length === 0) {
      this.taskBoxes = [];
      return;
    }

    const taskData = this.tasks.map((task) => {
      const start = new Date(task.created_at).getTime();
      const duration = (task.duration_seconds || 0) * 1000;
      return {
        id: task.id,
        name: task.name || task.id,
        start,
        duration,
        end: start + duration,
      };
    });

    this.minTime = Math.min(...taskData.map((t) => t.start));
    this.maxTime = Math.max(...taskData.map((t) => t.end));
    this.totalDuration = this.maxTime - this.minTime;

    if (this.totalDuration === 0) {
      this.taskBoxes = taskData.map((t) => ({
        id: t.id,
        name: t.name,
        left: 0,
        width: 100,
        startTime: t.start,
        duration: t.duration,
      }));
      return;
    }

    this.taskBoxes = taskData.map((t) => ({
      id: t.id,
      name: t.name,
      left: ((t.start - this.minTime) / this.totalDuration) * 100,
      width: (t.duration / this.totalDuration) * 100,
      startTime: t.start,
      duration: t.duration,
    }));
  }

  formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }
}
