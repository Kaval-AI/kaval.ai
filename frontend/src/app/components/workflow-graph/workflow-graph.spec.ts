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
import { WorkflowGraphComponent } from './workflow-graph';

describe('WorkflowGraphComponent', () => {
  let component: WorkflowGraphComponent;

  const workflow = {
    nodes: [
      { name: 'start', type: 'start', next: 'classify' },
      { name: 'classify', type: 'llm', next: 'route' },
      {
        name: 'route',
        type: 'switch',
        cases: { refund: 'refund_reply' },
        default: 'general',
      },
      { name: 'check', type: 'if', then: 'a', else_: 'b' },
      { name: 'refund_reply', type: 'agent', next: 'finish' },
      { name: 'finish', type: 'end' },
    ],
  };

  beforeEach(() => {
    component = new WorkflowGraphComponent();
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  it('builds a Mermaid graph with typed nodes and edges', () => {
    const def = component.buildDefinition(workflow);
    expect(def).toContain('graph TD');
    expect(def).toContain('n_start(["start"]):::startNode');
    expect(def).toContain(':::llmNode');
    expect(def).toContain('n_start --> n_classify');
    // switch fans out to its cases and default
    expect(def).toContain('n_route -->|refund| n_refund_reply');
    expect(def).toContain('n_route -->|default| n_general');
    // if branches to then/else
    expect(def).toContain('n_check -->|true| n_a');
    expect(def).toContain('n_check -->|false| n_b');
    // end node has no outgoing edge
    expect(def).not.toContain('n_finish -->');
  });

  it('handles a missing or empty workflow', () => {
    expect(component.buildDefinition(null)).toContain('graph TD');
    expect(component.buildDefinition({})).toContain('graph TD');
  });

  it('sanitizes node names into safe ids', () => {
    const def = component.buildDefinition({
      nodes: [{ name: 'my node!', type: 'llm', next: 'end node' }],
    });
    expect(def).toContain('n_my_node_');
    expect(def).toContain('n_end_node');
  });
});
