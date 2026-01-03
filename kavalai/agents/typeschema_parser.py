from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, create_model, Field


class TypeSchemaParser:
    def __init__(self):
        # Map TypeSchema types to Python types
        self.type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "any": Any,
        }

    def parse_schema(
        self, schema: Dict[str, Any], model_name: str = "GeneratedModel"
    ) -> Type[BaseModel]:
        """
        Recursively parses a TypeSchema and returns a Pydantic model.
        """
        fields = {}

        properties = schema.get("properties", {}) or schema.get("fields", {})
        required_fields = schema.get("required", [])

        for field_name, field_def in properties.items():
            ts_type = field_def.get("type", "any")

            # 1. Handle Nested Objects (Recursive)
            if ts_type == "object" and "properties" in field_def:
                nested_model = self.parse_schema(
                    field_def, model_name=field_name.capitalize()
                )
                python_type = nested_model

            # 2. Handle Arrays
            elif ts_type == "array":
                items = field_def.get("items", {})
                item_type = self.type_map.get(items.get("type", "any"), Any)

                # Recursive call if array items are objects
                if items.get("type") == "object":
                    item_type = self.parse_schema(items, model_name=f"{field_name}Item")

                python_type = List[item_type]
            # 3. Handle Primitives
            else:
                python_type = self.type_map.get(ts_type, Any)

            # 4. Handle Optionality
            if field_name not in required_fields:
                python_type = Optional[python_type]
                default_value = None
            else:
                default_value = ...  # Ellipsis means required in Pydantic

            # Add to field definitions
            fields[field_name] = (
                python_type,
                Field(default=default_value, description=field_def.get("description")),
            )

        # Create the Pydantic model dynamically
        return create_model(model_name, **fields)
