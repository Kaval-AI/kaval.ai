import { ComponentFixture, TestBed } from '@angular/core/testing';
import { JsonTreeComponent } from './json-tree';
import { By } from '@angular/platform-browser';

describe('JsonTreeComponent', () => {
  let component: JsonTreeComponent;
  let fixture: ComponentFixture<JsonTreeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [JsonTreeComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(JsonTreeComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    component.data = { key: 'value' };
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should format simple types correctly', () => {
    component.data = "hello";
    component.ngOnInit();
    expect(component.formattedValue).toBe('"hello"');
    expect(component.type).toBe('string');

    component.data = 123;
    component.ngOnInit();
    expect(component.formattedValue).toBe('123');
    expect(component.type).toBe('number');

    component.data = true;
    component.ngOnInit();
    expect(component.formattedValue).toBe('true');
    expect(component.type).toBe('boolean');

    component.data = null;
    component.ngOnInit();
    expect(component.formattedValue).toBe('null');
    expect(component.type).toBe('null');
  });

  it('should format arrays and objects correctly', () => {
    component.data = [1, 2, 3];
    component.ngOnInit();
    expect(component.formattedValue).toBe('Array[3]');
    expect(component.isObject).toBeTrue();
    expect(component.isArray).toBeTrue();
    expect(component.children.length).toBe(3);

    component.data = { a: 1, b: 2 };
    component.ngOnInit();
    expect(component.formattedValue).toBe('Object');
    expect(component.isObject).toBeTrue();
    expect(component.isArray).toBeFalse();
    expect(component.children.length).toBe(2);
  });

  it('should toggle expansion', () => {
    component.data = { a: 1 };
    component.ngOnInit();
    expect(component.isExpandable).toBeTrue();
    expect(component.isExpanded).toBeFalse();

    const event = new MouseEvent('click');
    spyOn(event, 'stopPropagation');

    component.toggleExpand(event);
    expect(component.isExpanded).toBeTrue();
    expect(event.stopPropagation).toHaveBeenCalled();

    component.toggleExpand(event);
    expect(component.isExpanded).toBeFalse();
  });

  it('should render children when expanded', () => {
    component.data = { key: 'value' };
    component.isExpanded = true;
    component.ngOnInit();
    fixture.detectChanges();

    const childElements = fixture.debugElement.queryAll(By.directive(JsonTreeComponent));
    expect(childElements.length).toBe(1);
    expect(childElements[0].componentInstance.data).toBe('value');
    expect(childElements[0].componentInstance.key).toBe('key');
  });
});
