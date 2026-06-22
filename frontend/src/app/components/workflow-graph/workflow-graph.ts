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
import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { AgentService } from '../../services/agent-service';

/**
 * Renders a workflow graph (its `nodes` + transitions) as an SVG diagram. The
 * SVG is generated on the backend from the agent's stored `workflow` JSON
 * (`kavalai.workflow.render_workflow_svg`), so node names, the arrows that
 * connect them and branch conditions are drawn consistently with the rest of
 * Kaval.AI — the documentation and the SDK use the same renderer.
 */
@Component({
  selector: 'app-workflow-graph',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (error) {
      <div class="text-error text-sm p-4">{{ error }}</div>
    }
    <div
      class="workflow-graph w-full flex justify-center overflow-auto"
      [innerHTML]="svg"
    ></div>
  `,
})
export class WorkflowGraphComponent implements OnChanges {
  @Input() workflow: any;
  svg: SafeHtml | null = null;
  error: string | null = null;

  constructor(
    private agentService: AgentService,
    private sanitizer: DomSanitizer,
  ) {}

  ngOnChanges(): void {
    this.render();
  }

  private render(): void {
    this.error = null;
    this.svg = null;
    if (!this.workflow) return;
    this.agentService.renderWorkflowSvg(this.workflow).subscribe({
      next: (svg) => {
        this.svg = this.sanitizer.bypassSecurityTrustHtml(svg);
      },
      error: () => {
        this.error = 'Could not render workflow graph.';
      },
    });
  }
}
