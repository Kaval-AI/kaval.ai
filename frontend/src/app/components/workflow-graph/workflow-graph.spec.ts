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
import { of, throwError } from 'rxjs';
import { WorkflowGraphComponent } from './workflow-graph';

describe('WorkflowGraphComponent', () => {
  let component: WorkflowGraphComponent;
  let agentService: { renderWorkflowSvg: jasmine.Spy };
  let sanitizer: { bypassSecurityTrustHtml: jasmine.Spy };

  const workflow = {
    nodes: [
      { name: 'start', type: 'start', next: 'reply' },
      { name: 'reply', type: 'llm', next: 'end' },
      { name: 'end', type: 'end' },
    ],
  };

  beforeEach(() => {
    agentService = { renderWorkflowSvg: jasmine.createSpy('renderWorkflowSvg') };
    sanitizer = {
      bypassSecurityTrustHtml: jasmine
        .createSpy('bypassSecurityTrustHtml')
        .and.callFake((value: string) => value),
    };
    component = new WorkflowGraphComponent(
      agentService as any,
      sanitizer as any,
    );
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  it('renders the backend-generated SVG for a workflow', () => {
    agentService.renderWorkflowSvg.and.returnValue(of('<svg>graph</svg>'));
    component.workflow = workflow;
    component.ngOnChanges();
    expect(agentService.renderWorkflowSvg).toHaveBeenCalledWith(workflow);
    expect(sanitizer.bypassSecurityTrustHtml).toHaveBeenCalledWith('<svg>graph</svg>');
    expect(component.svg).toBe('<svg>graph</svg>');
    expect(component.error).toBeNull();
  });

  it('shows an error when the backend render fails', () => {
    agentService.renderWorkflowSvg.and.returnValue(
      throwError(() => new Error('boom')),
    );
    component.workflow = workflow;
    component.ngOnChanges();
    expect(component.error).toContain('Could not render');
    expect(component.svg).toBeNull();
  });

  it('does nothing without a workflow', () => {
    component.workflow = null;
    component.ngOnChanges();
    expect(agentService.renderWorkflowSvg).not.toHaveBeenCalled();
    expect(component.svg).toBeNull();
  });
});
