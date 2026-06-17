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
import {
  AfterViewInit,
  Component,
  ElementRef,
  Input,
  OnChanges,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import mermaid from 'mermaid';

let mermaidInitialized = false;
let renderCounter = 0;

/**
 * Renders a v2 workflow graph (its `nodes` + transitions) as a Mermaid DAG.
 * Replaces the old backend-generated SVG: the diagram is built client-side from
 * the agent's stored `workflow` JSON.
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
      #container
      class="workflow-graph w-full flex justify-center overflow-auto"
    ></div>
  `,
})
export class WorkflowGraphComponent implements OnChanges, AfterViewInit {
  @Input() workflow: any;
  @ViewChild('container', { static: false })
  container?: ElementRef<HTMLDivElement>;
  error: string | null = null;

  async ngOnChanges(): Promise<void> {
    await this.render();
  }

  async ngAfterViewInit(): Promise<void> {
    await this.render();
  }

  /** Mermaid-safe id derived from a node name. */
  private nodeId(name: string): string {
    return 'n_' + String(name).replace(/[^a-zA-Z0-9_]/g, '_');
  }

  private escape(value: any): string {
    return String(value ?? '').replace(/"/g, "'");
  }

  /** Build the Mermaid `graph TD` definition from a v2 workflow JSON. */
  buildDefinition(workflow: any): string {
    const nodes: any[] = workflow?.nodes ?? [];
    const lines: string[] = ['graph TD'];

    for (const node of nodes) {
      const id = this.nodeId(node.name);
      const label = this.escape(node.name);
      switch (node.type) {
        case 'start':
          lines.push(`  ${id}(["${label}"]):::startNode`);
          break;
        case 'end':
          lines.push(`  ${id}(["${label}"]):::endNode`);
          break;
        case 'if':
        case 'switch':
          lines.push(`  ${id}{"${label}"}:::${node.type}Node`);
          break;
        default:
          lines.push(`  ${id}["${label}<br/>(${node.type})"]:::${node.type}Node`);
      }
    }

    for (const node of nodes) {
      const id = this.nodeId(node.name);
      if (node.type === 'end') continue;
      if (node.type === 'if') {
        if (node.then) lines.push(`  ${id} -->|true| ${this.nodeId(node.then)}`);
        const elseTarget = node.else_ ?? node.else;
        if (elseTarget)
          lines.push(`  ${id} -->|false| ${this.nodeId(elseTarget)}`);
      } else if (node.type === 'switch') {
        const cases = node.cases ?? {};
        for (const key of Object.keys(cases)) {
          lines.push(`  ${id} -->|${this.escape(key)}| ${this.nodeId(cases[key])}`);
        }
        if (node.default)
          lines.push(`  ${id} -->|default| ${this.nodeId(node.default)}`);
      } else if (node.next) {
        lines.push(`  ${id} --> ${this.nodeId(node.next)}`);
      }
    }

    lines.push('  classDef startNode fill:#16a34a,stroke:#15803d,color:#fff;');
    lines.push('  classDef endNode fill:#dc2626,stroke:#b91c1c,color:#fff;');
    lines.push('  classDef llmNode fill:#2563eb,stroke:#1d4ed8,color:#fff;');
    lines.push('  classDef agentNode fill:#7c3aed,stroke:#6d28d9,color:#fff;');
    lines.push('  classDef functionNode fill:#0891b2,stroke:#0e7490,color:#fff;');
    lines.push('  classDef ifNode fill:#d97706,stroke:#b45309,color:#fff;');
    lines.push('  classDef switchNode fill:#d97706,stroke:#b45309,color:#fff;');
    return lines.join('\n');
  }

  private async render(): Promise<void> {
    if (!this.workflow || !this.container) return;
    if (!mermaidInitialized) {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
      });
      mermaidInitialized = true;
    }
    try {
      const definition = this.buildDefinition(this.workflow);
      const { svg } = await mermaid.render(`wf-graph-${renderCounter++}`, definition);
      this.container.nativeElement.innerHTML = svg;
      this.error = null;
    } catch {
      this.error = 'Could not render workflow graph.';
    }
  }
}
