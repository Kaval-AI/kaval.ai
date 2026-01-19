import { Component, Input, OnInit, ViewChildren, QueryList } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-json-tree',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './json-tree.html',
  styleUrl: './json-tree.css'
})
export class JsonTreeComponent implements OnInit {
  @Input() data: any;
  @Input() key: string | number | null = null;
  @Input() isExpanded: boolean = false;
  @Input() depth: number = 0;

  @ViewChildren(JsonTreeComponent) childComponents!: QueryList<JsonTreeComponent>;

  isObject: boolean = false;
  isArray: boolean = false;
  isExpandable: boolean = false;
  formattedValue: string = '';
  type: string = '';

  children: { key: string | number, value: any }[] = [];

  ngOnInit() {
    this.isObject = false;
    this.isArray = false;
    this.isExpandable = false;
    this.children = [];

    this.type = typeof this.data;
    if (this.data === null) {
      this.type = 'null';
      this.formattedValue = 'null';
    } else if (Array.isArray(this.data)) {
      this.isArray = true;
      this.isObject = true;
      this.isExpandable = this.data.length > 0;
      this.formattedValue = `Array[${this.data.length}]`;
      this.children = this.data.map((value, index) => ({ key: index, value }));
    } else if (this.type === 'object') {
      this.isObject = true;
      const keys = Object.keys(this.data);
      this.isExpandable = keys.length > 0;
      this.formattedValue = 'Object';
      this.children = keys.map(key => ({ key, value: this.data[key] }));
    } else if (this.type === 'string') {
      this.formattedValue = `"${this.data}"`;
    } else {
      this.formattedValue = String(this.data);
    }
  }

  toggleExpand(event: Event) {
    if (this.isExpandable) {
      this.isExpanded = !this.isExpanded;
    }
    event.stopPropagation();
  }

  expandAll(event: Event) {
    this.setExpandedRecursive(true);
    event.stopPropagation();
  }

  collapseAll(event: Event) {
    this.setExpandedRecursive(false);
    event.stopPropagation();
  }

  private setExpandedRecursive(expanded: boolean) {
    if (this.isExpandable) {
      this.isExpanded = expanded;
      setTimeout(() => {
        this.childComponents.forEach(child => child.setExpandedRecursive(expanded));
      });
    }
  }
}
