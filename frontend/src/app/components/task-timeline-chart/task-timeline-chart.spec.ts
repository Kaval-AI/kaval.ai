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

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { TaskTimelineChart } from './task-timeline-chart';
import { Task } from '../../models/task';
import { SimpleChange } from '@angular/core';

describe('TaskTimelineChart', () => {
  let component: TaskTimelineChart;
  let fixture: ComponentFixture<TaskTimelineChart>;

  const mockTasks: Task[] = [
    {
      id: 'task-1',
      name: 'Task 1',
      created_at: '2026-03-26T00:00:00Z',
      duration_seconds: 10,
    } as Task,
    {
      id: 'task-2',
      name: 'Task 2',
      created_at: '2026-03-26T00:00:10Z',
      duration_seconds: 20,
    } as Task,
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TaskTimelineChart],
    }).compileComponents();

    fixture = TestBed.createComponent(TaskTimelineChart);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should calculate task boxes correctly', () => {
    component.tasks = mockTasks;
    component.ngOnChanges({
      tasks: new SimpleChange(null, mockTasks, true),
    });

    expect(component.taskBoxes.length).toBe(2);

    // Total duration is 10s (task 1) + 20s (task 2 starting at 10s) = 30s
    expect(component.totalDuration).toBe(30000);

    // Task 1: starts at 0, duration 10s (1/3 of total)
    expect(component.taskBoxes[0].left).toBe(0);
    expect(component.taskBoxes[0].width).toBeCloseTo(33.33, 1);

    // Task 2: starts at 10s (1/3 of total), duration 20s (2/3 of total)
    expect(component.taskBoxes[1].left).toBeCloseTo(33.33, 1);
    expect(component.taskBoxes[1].width).toBeCloseTo(66.66, 1);
  });

  it('should handle empty tasks', () => {
    component.tasks = [];
    component.ngOnChanges({
      tasks: new SimpleChange(null, [], true),
    });

    expect(component.taskBoxes.length).toBe(0);
  });

  it('should format duration correctly', () => {
    expect(component.formatDuration(500)).toBe('500ms');
    expect(component.formatDuration(1500)).toBe('1.50s');
    expect(component.formatDuration(60000)).toBe('60.00s');
  });
});
